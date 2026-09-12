#!/usr/bin/env bash
# One-shot migration from per-run timers to the single tick. Runs ON THE VPS,
# as root. Idempotent: run it twice and the second pass finds nothing to remove.
#
# What it does, in order:
#   1. disables and removes every quantlab-poll@<id>.timer instance and the
#      legacy single-run quantlab-poll.timer / quantlab-poll.service pair
#   2. installs quantlab-tick.timer, quantlab-tick.service, quantlab-book.service
#      and the updated quantlab-poll@.service from deploy/
#   3. generates the Wants= drop-in (deploy/sync-tick.sh)
#   4. enables quantlab-tick.timer and prints list-timers
#
# It never touches paper_runs/. A poll that is mid-flight when this runs is left
# to finish — the template service is replaced, not stopped.
#
#     sudo deploy/migrate-to-tick.sh
#     sudo QUANTLAB_ROOT=/home/me/tradingview-mcp deploy/migrate-to-tick.sh
set -euo pipefail

if [ "$(uname -s)" != "Linux" ] || ! command -v systemctl >/dev/null 2>&1; then
    echo "migrate-to-tick: this targets the VPS (Linux + systemd); nothing to do here" >&2
    exit 1
fi
if [ "$(id -u)" -ne 0 ]; then
    echo "migrate-to-tick: run as root (sudo) — it writes /etc/systemd/system" >&2
    exit 1
fi

ROOT="${QUANTLAB_ROOT:-/root/tradingview-mcp}"
HERE="$(cd "$(dirname "$0")" && pwd)"
UNITS=/etc/systemd/system

echo "== 1. per-run timers and the legacy single-run pair"
# instances that are enabled show up as symlinks under timers.target.wants; ones
# that were only ever started show up in list-units. Cover both, then the files.
for t in $(systemctl list-units --all --plain --no-legend 'quantlab-poll@*.timer' 'quantlab-poll.timer' 2>/dev/null | awk '{print $1}'); do
    echo "   disable --now $t"
    systemctl disable --now "$t" 2>/dev/null || systemctl stop "$t" 2>/dev/null || true
done
for link in "$UNITS"/timers.target.wants/quantlab-poll@*.timer "$UNITS"/timers.target.wants/quantlab-poll.timer; do
    [ -e "$link" ] || [ -L "$link" ] || continue
    echo "   rm $link"
    rm -f "$link"
done
# the legacy single-run service may be running right now; let it finish, then drop it
if systemctl is-active --quiet quantlab-poll.service 2>/dev/null; then
    echo "   waiting for the legacy quantlab-poll.service to finish its poll"
    while systemctl is-active --quiet quantlab-poll.service; do sleep 5; done
fi
for f in "$UNITS"/quantlab-poll@.timer "$UNITS"/quantlab-poll.timer "$UNITS"/quantlab-poll.service; do
    [ -e "$f" ] || continue
    echo "   rm $f"
    rm -f "$f"
done

echo "== 2. install the tick units"
for f in quantlab-tick.timer quantlab-tick.service quantlab-book.service quantlab-poll@.service; do
    echo "   $f"
    install -m 644 "$HERE/$f" "$UNITS/$f"
done
systemctl daemon-reload

echo "== 3. Wants= drop-in"
QUANTLAB_ROOT="$ROOT" "$HERE/sync-tick.sh"

echo "== 4. enable the tick"
systemctl enable --now quantlab-tick.timer
echo
systemctl list-timers 'quantlab-*' --no-pager
echo
echo "Next tick will poll every run listed above, then size the book. Prove it now:"
echo "    systemctl start quantlab-tick.service"
echo "    journalctl -u 'quantlab-*' --since '-5 min' --no-pager -o short-iso"
