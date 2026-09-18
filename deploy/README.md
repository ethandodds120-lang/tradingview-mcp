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
| `quantlab-heartbeat.timer` / `.service` | its own clock, `*:2/5`: `paper.py heartbeat` — says when a run or the book has gone stale (§8) |

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

## 8. Notice when it stops — the heartbeat

The failure mode is silence, not a crash. `state.json`'s `updated` field moves
at the end of every poll of every run, and `book.json`'s `as_of` moves on
every tick, so those two are the heartbeats. What reads them is
`paper.py heartbeat`, run by **its own timer** — `quantlab-heartbeat.timer`,
`*:2/5`, two minutes after each tick — and deliberately *not* by the tick's
`Wants=` list: a dead tick cannot report itself, so the watcher must not depend
on the thing it watches.

What it checks, every five minutes (DESIGN-risk.md §3):

| thing | fresh means | else |
|---|---|---|
| each run that is not `STOPPED` — halted runs included, they still poll | `state.updated` no older than **2 × tick_s** (`execution.tick_s`, else 300 → 600 s) | stale |
| `paper_runs/book.json` | `as_of` no older than 2 × tick_s | stale |

On a fresh → stale transition it writes one `ALERT … [stale] …` line to
`paper_runs/alerts.log` and pushes it to Telegram if §9 is set up; while the
thing stays stale it repeats **at most once every six hours**, so a dead
network is one message, not four an hour. Every stale item that is due goes
in **one message per pass**, so a pass makes at most one push. A push that
fails on the transport does not count as the six-hourly alert: the item stays
due, but it is not retried on every five-minute pass — the failure stamps
`retry_not_before` (15 minutes on) on the item in `heartbeat.json`, together
with `last_attempt` and `last_error`, and the pass that finally delivers it is
the one the six hours count from. `heartbeat.json` is written before the
push, so a pass that dies mid-push still leaves `stale_since` on disk. Each
`notify` is exactly one line in `alerts.log` — a multi-item message has its
newlines folded to ` | ` there (the Telegram text keeps them), and so is the
`[telegram] push ... failed` line that follows a failed push, whatever the
transport returned (an HTML error page from Telegram's edge is folded the same
way, and stored folded as the item's `last_error`). Recovery (stale → fresh) is
written to `alerts.log` and clears the item from `paper_runs/heartbeat.json`;
it is not pushed unless the unit's `ExecStart` carries `--push-recovery`.

Install the timer (both files are in `deploy/`; `User=` and the paths are set
for root's checkout at `/root/tradingview-mcp`, edit them as in §4 otherwise):

```bash
sudo cp ~/tradingview-mcp/deploy/quantlab-heartbeat.service \
        ~/tradingview-mcp/deploy/quantlab-heartbeat.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now quantlab-heartbeat.timer
systemctl list-timers 'quantlab-*'          # want: quantlab-tick.timer AND quantlab-heartbeat.timer
```

Then look before the timer does:

```bash
cd ~/tradingview-mcp
env/bin/python paper.py heartbeat --dry-run   # prints; writes nothing, sends nothing
```

Every run in `wants.conf` should show `fresh` with an age under 300 s and
`book.json` the same. Then one real pass, so the first alert (if any) happens
while you are watching rather than at 03:00:

```bash
env/bin/python paper.py heartbeat
tail -5 paper_runs/alerts.log
cat paper_runs/heartbeat.json
```

A stale book is not silent on the run side either: each decision made without a
fresh book prints an `ALERT … [book] …` line and appends it to
`paper_runs/alerts.log`. That one stays log-only — the heartbeat's `stale` on
`book.json` already carries the cause.

Limit, stated plainly: if systemd or the box itself is down, nothing on the box
can say so. An external check of the box is outside this repo.

The kill rules (DESIGN-risk.md §1) are the other thing that speaks up on its
own: a run that trips one writes a `HALTED` marker, journals it, pushes a
`halt` alert, and from then on polls and journals without deciding or routing
until a person runs `paper.py resume --id <run> --reason "..."`. `paper.py
risk --all` shows every rule's numbers, read-only, at any time. A halted run
is still in `wants.conf` (it must keep polling); only `paper.py stop` takes it
out. Run `resume` between ticks: a poll already in flight (or a foreground
`paper.py run` loop) re-reads the marker and the new epoch from disk before it
evaluates the rules, and its save keeps the epoch on disk whichever epoch it
was holding (none, or that of an earlier resume), but the tick that is
running while you type is the one that should not also be deciding.

## 9. Telegram — the push channel

Three events are pushed from the box: a routed fill, a stale run or book, and a
kill-rule halt (plus `alert --test`). Nothing else leaves the box, and nothing
is pushed by a chat session or anything that needs a person logged in. The
transport is one HTTPS POST to Telegram's `sendMessage` from the poll, the
book and the heartbeat units, 10 s timeout, one retry, **20 s of wall clock
at most per alert** — each attempt runs in a worker thread that is abandoned
at 10 s, because urllib's timeout bounds a socket operation, not DNS or a
server that drips bytes; a failure is one more line in `alerts.log`, never an
exception into a poll.

**Every step below is done by you, by hand. The environment file is written by
you and read by systemd. No code in this repo and no chat session ever reads,
writes, prints or transmits the token; `paper.py alert --test` reports only
whether the two variables are set and whether Telegram accepted a test line.**

1. **Create the bot.** In Telegram, message `@BotFather`, send `/newbot`, give
   it a name and a username ending in `bot`. BotFather replies with the bot
   token (`123456789:AA…`). Keep it; do not paste it into a chat with anyone,
   including an assistant.

2. **Get your chat id.** Open a chat with the new bot and send it any message
   (a bot cannot message you until you have messaged it). Then, in a browser
   or with curl **on your own machine**, open
   `https://api.telegram.org/bot<token>/getUpdates` and read the
   `"chat":{"id": …}` number from the reply. For a group, add the bot to the
   group, send a message there, and the id is the negative number in the same
   place.

3. **Write the environment file on the VPS, as root, mode 600.** Type it in an
   editor; do not build it with a script and do not put the token on a command
   line where it lands in shell history:

   ```bash
   sudo mkdir -p /etc/quantlab
   sudo chmod 700 /etc/quantlab
   sudo nano /etc/quantlab/telegram.env
   ```

   with exactly two lines:

   ```
   TELEGRAM_BOT_TOKEN=123456789:AA...
   TELEGRAM_CHAT_ID=987654321
   ```

   then

   ```bash
   sudo chmod 600 /etc/quantlab/telegram.env
   sudo chown root:root /etc/quantlab/telegram.env     # the units run as root
   ```

   If the units run as another user, `chown` the file to that user instead —
   `EnvironmentFile=` is read by systemd as the service's user.

4. **The units already point at it.** `quantlab-poll@.service`,
   `quantlab-book.service` and `quantlab-heartbeat.service` each carry
   `EnvironmentFile=-/etc/quantlab/telegram.env`; the leading `-` means a
   missing file is not an error, so a box without the file runs exactly as
   before and only logs. After editing units or installing them for the first
   time: `sudo systemctl daemon-reload`.

5. **Install the heartbeat timer** if §8 has not been done yet.

6. **Test it from the box, as the unit's user, with the file loaded the way
   systemd loads it:**

   ```bash
   cd ~/tradingview-mcp
   sudo systemd-run --wait --pipe --collect -p EnvironmentFile=/etc/quantlab/telegram.env \
       -p WorkingDirectory=/root/tradingview-mcp \
       /root/tradingview-mcp/env/bin/python paper.py alert --test
   ```

   or, more simply, run it as root from a shell that has read the file:

   ```bash
   sudo bash -c 'set -a; . /etc/quantlab/telegram.env; set +a; cd /root/tradingview-mcp && env/bin/python paper.py alert --test'
   ```

   It prints `TELEGRAM_BOT_TOKEN set` / `TELEGRAM_CHAT_ID set` (never the
   values), sends one fixed line naming the host and the UTC time, and says
   whether Telegram accepted it. With the variables unset it says so, sends
   nothing, tries no network, and exits 0.

7. **Rotate or revoke** with BotFather (`/revoke`), then rewrite the file by
   hand. Nothing caches the token: the next unit start reads the new one.
