# Deploying the forward test to a Linux VPS

The repo is only half of what the VPS needs. Two things are gitignored and will
**not** arrive with `git clone` — copy them by hand or the run cannot continue:

| what | why it matters |
|---|---|
| `.env` | the Alpaca keys. Never commit it. |
| `paper_runs/<run-id>/` | the run's state, bars and journal. Without it the VPS has no run to poll, and creating a fresh one restarts the forward test from zero. |

Everything below assumes run id `sol-trend-20260906`. Substitute your own.

## How the schedule works

One timer, one transaction. `quantlab-tick.timer` fires `quantlab-tick.service`
every five minutes; that unit does nothing itself but *wants* one
`quantlab-poll@<id>.service` per run plus `quantlab-book.service`, and the polls
are ordered `Before=` the book. So every tick is: all polls, then the book sizes
the whole book off the state the polls just wrote and publishes
`paper_runs/book.json`. Runs read that on their next decision.

Not every decision is sent by the tick that makes it. A crypto run works its
order on the spot. An equity run decides after the close — the 16:05 ET tick
sees the completed daily bar — but only *plans* the order then; Alpaca accepts
market-on-open (OPG) orders in a window, so a later tick sends it, some time
between 19:00 ET and 09:28 ET the next morning, and it fills at the open. If
the box is down for that whole window the run has no OPG order in; the first
tick after the open sends a plain market order instead and marks the fill
`late` in the journal, so the delay shows up in `report --execution` rather
than passing for a fill at the open.

The list of runs the tick wants is a generated drop-in,
`/etc/systemd/system/quantlab-tick.service.d/wants.conf`, produced by
`deploy/sync-tick.sh` from `paper.py tick-wants`. It lists every run under
`paper_runs/` that has no `STOPPED` marker. **A run that is not in that file is
not polled**, which is why every `start` and `stop` is followed by a sync.

| unit | what |
|---|---|
| `quantlab-tick.timer` | the clock: `*:0/5`, persistent, 5s jitter |
| `quantlab-tick.service` | `/bin/true` + the generated `Wants=` drop-in |
| `quantlab-poll@.service` | one poll of run `%i`; `Before=quantlab-book.service` |
| `quantlab-book.service` | `paper.py book --write`, after the polls |

The old per-run `quantlab-poll@<id>.timer` instances and the single-run
`quantlab-poll.service`/`.timer` pair are superseded; `migrate-to-tick.sh`
removes them. That script rewrites `/etc/systemd/system` and is run once, by
you, after you have read it — it is not something an agent or a deploy hook
runs on its own.

A run id doubles as the systemd instance name, so `paper.py start` refuses one
that does not match `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`, and `tick-wants`
leaves any run directory that fails the same check out of `Wants=` with a
warning on stderr. Do not rename a run directory by hand into something the
tick cannot name.

## 1. On the VPS

```bash
sudo apt update && sudo apt install -y python3 python3-venv git
git clone https://github.com/ethandodds120-lang/tradingview-mcp.git
cd tradingview-mcp
python3 -m venv env
env/bin/pip install -r requirements.txt
```

Set the clock to UTC. Bars are stored tz-naive UTC and a box on local time will
misalign the daily close:

```bash
sudo timedatectl set-timezone UTC
```

## 2. Copy the two gitignored things across

Run these **from the Windows machine**, not the VPS:

```powershell
scp .env YOUR_USER@VPS_IP:~/tradingview-mcp/.env
scp -r paper_runs/sol-trend-20260906 YOUR_USER@VPS_IP:~/tradingview-mcp/paper_runs/
```

Then lock the keys down, on the VPS:

```bash
chmod 600 ~/tradingview-mcp/.env
```

## 3. Verify before scheduling anything

```bash
cd ~/tradingview-mcp
env/bin/python paper.py broker --symbol "SOL/USD"
env/bin/python paper.py report --id sol-trend-20260906
```

The broker check must say `paper` and name account `PA3CMIZW2AHR`. The report must
show the forward bars and fills you already have — if it says 0 forward bars, the
`paper_runs/` copy did not land and you are about to start a new experiment rather
than continue the existing one. Stop and fix that first.

Then one manual poll, to prove the whole path works while you are watching, and
one read-only book, to see what the tick will publish:

```bash
env/bin/python paper.py poll --id sol-trend-20260906
env/bin/python paper.py book
```

## 4. Install the tick

The shipped units are written for `root`, whose checkout lives at
`/root/tradingview-mcp`. If you run as a normal user, edit `User=` and both paths
in `quantlab-poll@.service` and `quantlab-book.service` first, and pass
`QUANTLAB_ROOT=/home/<you>/tradingview-mcp` to the scripts — and do not do it with
a blind find-and-replace, because `/home/root` is not root's home and the
resulting failure is an unexplained `status=200/CHDIR`.

Fresh box, or a box still on the per-run timers — either way, the migration
script is the install. It is idempotent and removes nothing under `paper_runs/`.
Read it before you run it, and run it yourself: it disables timers and replaces
unit files under `/etc/systemd/system`, which is not a step to delegate.

```bash
less ~/tradingview-mcp/deploy/migrate-to-tick.sh
sudo ~/tradingview-mcp/deploy/migrate-to-tick.sh
```

It prints `list-timers` at the end; you want one line, `quantlab-tick.timer`, and
no `quantlab-poll@*.timer`. Then prove a tick works rather than waiting five
minutes to find out:

```bash
sudo systemctl start quantlab-tick.service
journalctl -u 'quantlab-*' --since '-5 min' --no-pager -o short-iso
systemctl show quantlab-book.service -p Result --value     # want: success
cat ~/tradingview-mcp/paper_runs/book.json | head -20
```

The journal should show every `quantlab-poll@<id>` finishing before
`quantlab-book` starts. If the book ran first, a poll is missing
`Before=quantlab-book.service` — you installed an old unit file.

## 5. Adding a run (and stopping one)

```bash
cd ~/tradingview-mcp
env/bin/python paper.py start --strategy tsmom --alpaca QQQ --id qqq-tsmom-20260912 \
    && sudo deploy/sync-tick.sh
```

`sync-tick.sh` rewrites `wants.conf` and reloads systemd; it prints what the tick
now wants so you can see the new run in the list. Nothing else changes — the
template service and the timer are the same files for every run.

Stopping is the mirror image. `paper.py stop` writes the `STOPPED` marker and the
run drops out of the next sync; its journal, bars and state stay on disk:

```bash
env/bin/python paper.py stop --id qqq-tsmom-20260912 --reason "why" \
    && sudo deploy/sync-tick.sh
```

Forgetting the sync is the one failure this design has: the run exists, `list`
shows it, and it is never polled. Check `wants.conf` when in doubt:

```bash
cat /etc/systemd/system/quantlab-tick.service.d/wants.conf
```

## 6. Reboot test

Do this once after install and once after any change to the units. The point is
not that the box comes back — it is that the schedule comes back *with all runs*
and in the right order, without anyone logging in.

```bash
sudo systemctl reboot
```

After it is up again, in this order:

1. **The timer is armed.** `systemctl list-timers quantlab-tick.timer` shows a
   `NEXT` within five minutes. Because the timer is `Persistent=true`, a tick
   that was missed while the box was down fires immediately on boot.
2. **One tick ran, polls before book.** Within one tick of boot:

   ```bash
   journalctl -u 'quantlab-*' --since '-10 min' --no-pager -o short-iso
   ```

   Every `quantlab-poll@<id>` in `wants.conf` appears, each `Finished` before
   `Starting quantlab-book`. A run missing here is a run missing from
   `wants.conf`; a book that starts first is an ordering bug.
3. **Every run's heartbeat moved.** `state.updated` is written at the end of every
   poll, new bar or not, so all of them must be newer than the boot:

   ```bash
   cd ~/tradingview-mcp && for s in paper_runs/*/state.json; do
       printf '%-40s %s\n' "$s" "$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['updated'])" "$s")"
   done
   uptime -s
   ```

   A `STOPPED` run is allowed to be stale; anything else older than the boot did
   not poll.
4. **The book is fresh.** `paper_runs/book.json`'s `as_of` is newer than the boot,
   and `applies_to` lists every non-stopped run.

If any of the four fails, the box is not deployed, however healthy it looks.

## 7. Only one machine may poll a run

The run's state lives in `paper_runs/<id>/state.json` on whichever machine polls
it. Two machines polling the same run keep two divergent copies of that state and
both send orders to the same Alpaca account — double fills, and a journal that
matches neither.

Before enabling the timer, make sure nothing on the Windows box is still polling:

```powershell
Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*paper.py*" }
Get-ScheduledTask -TaskName "quantlab-sol-trend" -ErrorAction SilentlyContinue
```

Both should come back empty. The VPS is now the only writer.

## 8. Notice when it stops

The failure mode is silence, not a crash. `state.json`'s `updated` field is the
heartbeat for each run and `book.json`'s `as_of` is the heartbeat for the tick.
With the tick at five minutes, anything over about fifteen minutes stale means
the schedule is not running:

```bash
python3 - <<'PY'
import json, datetime, pathlib
now = datetime.datetime.now(datetime.UTC)
for p in sorted(pathlib.Path("paper_runs").glob("*/state.json")):
    if (p.parent / "STOPPED").exists():
        continue
    age = now - datetime.datetime.fromisoformat(json.loads(p.read_text())["updated"])
    print(f"{p.parent.name:<32} last poll {age.total_seconds()/60:5.1f} min ago",
          "STALE" if age.total_seconds() > 900 else "ok")
b = pathlib.Path("paper_runs/book.json")
if b.exists():
    age = now - datetime.datetime.fromisoformat(json.loads(b.read_text())["as_of"])
    print(f"{'book':<32} as_of     {age.total_seconds()/60:5.1f} min ago",
          "STALE" if age.total_seconds() > 900 else "ok")
else:
    print("book.json missing — every run is sizing at k=1 and saying so in alerts.log")
PY
```

A stale book is not silent on the run side either: each decision made without a
fresh book prints an `ALERT` line and appends it to `paper_runs/alerts.log`. Tail
that file too.

Wire this into whatever alerting you use. Without it you will not find out the bot
died until you go looking, which last time was three days.
