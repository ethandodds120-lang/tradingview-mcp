# Deploying the forward test to a Linux VPS

The repo is only half of what the VPS needs. Two things are gitignored and will
**not** arrive with `git clone` — copy them by hand or the run cannot continue:

| what | why it matters |
|---|---|
| `.env` | the Alpaca keys. Never commit it. |
| `paper_runs/<run-id>/` | the run's state, bars and journal. Without it the VPS has no run to poll, and creating a fresh one restarts the forward test from zero. |

Everything below assumes run id `sol-trend-20260906`. Substitute your own.

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

Then one manual poll, to prove the whole path works while you are watching:

```bash
env/bin/python paper.py poll --id sol-trend-20260906
```

## 4. Install the timer

The shipped unit is written for `root`, whose checkout lives at
`/root/tradingview-mcp`. If you run as a normal user, edit `User=` and both paths
to `/home/<you>/tradingview-mcp` first — and do not do it with a blind
find-and-replace, because `/home/root` is not root's home and the resulting
failure is an unexplained `status=200/CHDIR`.

```bash
sudo cp ~/tradingview-mcp/deploy/quantlab-poll.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now quantlab-poll.timer
```

Prove the unit works rather than waiting an hour to find out:

```bash
sudo systemctl start quantlab-poll.service
systemctl show quantlab-poll.service -p Result --value   # want: success
journalctl -u quantlab-poll -n 10 --no-pager -o cat
```

Check it:

```bash
systemctl list-timers quantlab-poll.timer
journalctl -u quantlab-poll -n 50 --no-pager
```

## 5. Only one machine may poll a run

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

## 6. Notice when it stops

The failure mode is silence, not a crash. `state.json`'s `updated` field is the
heartbeat — if it is more than about two hours stale, the timer is not running:

```bash
python3 - <<'PY'
import json, datetime, pathlib
s = json.loads(pathlib.Path("paper_runs/sol-trend-20260906/state.json").read_text())
age = datetime.datetime.now(datetime.UTC) - datetime.datetime.fromisoformat(s["updated"])
print(f"last poll {age.total_seconds()/3600:.1f}h ago", "STALE" if age.total_seconds() > 7200 else "ok")
PY
```

Wire that into whatever alerting you use. Without it you will not find out the bot
died until you go looking, which last time was three days.
