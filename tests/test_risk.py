"""Tests for the kill rules, the halt marker, the heartbeat and the alerts
(DESIGN-risk.md §1–§4, the §5 list).

Runs standalone (`python tests/test_risk.py`) and under pytest. Run directories
are built by hand — config.json, state.json, journal.jsonl, bars.csv — so each
rule can be armed and tripped on a journal that says exactly what the test
wants it to say; the halt-by-poll tests use a real replay-feed run on synthetic
bars. No feed or broker reaches a network: the replay feed reads a file, and
the Telegram transport is replaced with a recorder. The environment variables
are set to fake values inside the alert tests and restored afterwards — no
test ever reads a real token.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
import traceback
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from quantlab import alerts, book, data, risk  # noqa: E402
from quantlab import paper as P  # noqa: E402

TMP = Path(os.environ.get("QUANTLAB_TEST_TMP") or tempfile.mkdtemp(prefix="quantlab-risk-"))
UTC = timezone.utc
NOW = datetime(2026, 9, 12, 12, 0, 0, tzinfo=UTC)


def _fresh(name: str) -> Path:
    base = TMP / name
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)
    return base


def _iso(ts: datetime) -> str:
    return ts.replace(microsecond=0).isoformat()


def _write_bars(path: Path, bars: pd.DataFrame) -> None:
    out = bars.reset_index()
    out.columns = ["time", *P.BAR_COLS]
    out.to_csv(path, index=False)


def _records(root: Path) -> list[dict]:
    text = (root / "journal.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def bar_rec(ts, equity: float, fill: dict | None = None, held: float = 0.0) -> dict:
    return {"type": "bar", "at": "2026-09-12T00:00:00+00:00", "bar": str(ts),
            "open": 100.0, "close": 100.0, "fill": fill, "equity": round(equity, 2),
            "held": held, "drawdown": 0.0, "signal": 1.0, "target": 0.5,
            "vol_scale": 0.5, "order": None}


def fill(side: str, units: float, price: float) -> dict:
    return {"side": side, "units": units if side == "buy" else -units,
            "ref_price": price, "fill_price": price, "notional": abs(units) * price,
            "commission": 0.0, "slippage_cost": 0.0, "decided_at": "x"}


def make_run(base: Path, run_id: str, forward: list[dict], equity: float,
             n_seed: int = 300, strategy: str = "buy_hold", vol_target=0.15,
             start_equity: float = 100_000.0, seed: int = 1,
             bars: pd.DataFrame | None = None) -> P.PaperRun:
    """A run directory built by hand. `forward` is the list of journal bar
    records after the forward start (their `bar` stamps are rewritten onto the
    synthetic index so they follow the seed history); `equity` is what state
    says now. `bars` replaces the synthetic daily frame (its last len(forward)
    bars are the forward ones)."""
    if bars is None:
        bars = data.synthetic(n=n_seed + len(forward), seed=seed)
    else:
        n_seed = len(bars) - len(forward)
    root = base / run_id
    root.mkdir()
    _write_bars(root / "bars.csv", bars)
    fwd_idx = bars.index[n_seed:]
    for rec, ts in zip(forward, fwd_idx):
        rec["bar"] = str(ts)
    forward_start = str(bars.index[n_seed - 1])
    last_bar = str(fwd_idx[-1]) if len(fwd_idx) else forward_start
    config = {"run_id": run_id, "created": "2026-09-01T00:00:00+00:00", "strategy": strategy,
              "params": dict(P.REGISTRY[strategy].params),
              "feed": {"kind": "csv", "path": str(root / "bars.csv")}, "broker": None,
              "cost_bps": 3.0, "vol_target": vol_target, "vol_lookback": 60,
              "max_leverage": 2.0, "rebalance_band": 0.10, "start_equity": start_equity,
              "min_history": 200, "note": "test"}
    config["fingerprint"] = P.fingerprint(config)
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    state = {"last_bar": last_bar, "forward_start": forward_start, "cash": equity,
             "units": 0.0, "pending": None, "equity": equity, "peak_equity": start_equity,
             "bars_seen": n_seed + len(forward), "bars_forward": len(forward),
             "fills": sum(1 for r in forward if r.get("fill")), "updated": _iso(NOW)}
    (root / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    with (root / "journal.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": "created", "at": config["created"], "config": config}) + "\n")
        fh.write(json.dumps({"type": "backfill", "at": config["created"], "bar": forward_start,
                             "history_bars": n_seed}) + "\n")
        for rec in forward:
            fh.write(json.dumps(rec) + "\n")
    return P.PaperRun.load(base, run_id)


def losing_trades(n: int, equity: float = 100_000.0) -> list[dict]:
    """n closed round trips, every one a loss of 10.0: buy 10 @ 100, sell 10 @ 99."""
    out = []
    for _ in range(n):
        out.append(bar_rec(None, equity, fill("buy", 10.0, 100.0), held=0.01))
        out.append(bar_rec(None, equity, fill("sell", 10.0, 99.0), held=0.0))
    return out


def replay_run(base: Path, run_id: str, n: int = 400, start: int = 250,
               strategy: str = "buy_hold", seed: int = 3) -> P.PaperRun:
    """A real run on the replay feed over synthetic bars — no network anywhere."""
    csv = base / f"{run_id}-bars.csv"
    _write_bars(csv, data.synthetic(n=n, seed=seed))
    return P.PaperRun.create(base, run_id, strategy,
                             {"kind": "replay", "path": str(csv), "start": start, "step": 1},
                             min_history=100, vol_lookback=60)


class Recorder:
    """A fake Telegram transport: records every call, optionally fails."""

    def __init__(self, fail: Exception | None = None, status: int = 200):
        self.calls, self.fail, self.status = [], fail, status

    def __call__(self, url, body, timeout):
        self.calls.append({"url": url, "body": body, "timeout": timeout})
        if self.fail:
            raise self.fail
        return self.status, json.dumps({"ok": True})


class Env:
    """Set the two Telegram variables for one block, then put them back."""

    def __init__(self, token=None, chat=None):
        self.values = {alerts.TOKEN_VAR: token, alerts.CHAT_VAR: chat}

    def __enter__(self):
        self.saved = {k: os.environ.get(k) for k in self.values}
        for k, v in self.values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _quiet_alerts():
    """The tests that halt run with Telegram unset and a transport that would
    fail loudly if anything reached it."""
    rec = Recorder(fail=AssertionError("transport must not be called"))
    alerts._post = rec
    return rec


# ────────────────────────────── R1 profit factor ──────────────────────────────

def test_r1_unarmed_below_60_trades_armed_and_trips_at_60():
    base = _fresh("r1")
    run = make_run(base, "r1-59", losing_trades(59), equity=100_000.0)
    ev = risk.evaluate(run)
    r1 = ev["rules"]["r1_profit_factor"]
    assert r1["value"] == 0.0 and not r1["armed"] and not r1["trip"], r1
    assert r1["inputs"]["closed_in_epoch"] == 59 and r1["inputs"]["trades_in_window"] == 59
    assert ev["tripped"] is None

    run = make_run(base, "r1-60", losing_trades(60), equity=100_000.0)
    ev = risk.evaluate(run)
    r1 = ev["rules"]["r1_profit_factor"]
    assert r1["armed"] and r1["trip"] and r1["value"] == 0.0 and r1["threshold"] == 1.0, r1
    assert r1["inputs"]["losses"] == 60 and r1["inputs"]["gross_loss"] == 600.0
    assert ev["tripped"] == "r1_profit_factor"
    # the other two are armed (120 epoch bars) but a flat equity trips neither
    assert ev["rules"]["r2_equity_band"]["armed"] and not ev["rules"]["r2_equity_band"]["trip"]
    assert ev["rules"]["r3_realised_vol"]["armed"] and not ev["rules"]["r3_realised_vol"]["trip"]


def test_r1_winners_and_no_losses():
    base = _fresh("r1b")
    trades = []
    for i in range(60):
        exit_px = 101.0 if i % 2 else 99.0            # 30 wins of 10, 30 losses of 10
        trades.append(bar_rec(None, 100_000.0, fill("buy", 10.0, 100.0)))
        trades.append(bar_rec(None, 100_000.0, fill("sell", 10.0, exit_px)))
    run = make_run(base, "r1-even", trades, equity=100_000.0)
    r1 = risk.evaluate(run)["rules"]["r1_profit_factor"]
    assert r1["armed"] and abs(r1["value"] - 1.0) < 1e-9 and not r1["trip"], r1
    # no losses at all: PF is +inf, armed, not tripped
    trades = []
    for _ in range(60):
        trades.append(bar_rec(None, 100_000.0, fill("buy", 10.0, 100.0)))
        trades.append(bar_rec(None, 100_000.0, fill("sell", 10.0, 101.0)))
    run = make_run(base, "r1-wins", trades, equity=100_000.0)
    ev = risk.evaluate(run)
    r1 = ev["rules"]["r1_profit_factor"]
    # stored as None + no_losses, never as an `Infinity` token in the JSON
    assert r1["armed"] and r1["value"] is None and not r1["trip"], r1
    assert r1["inputs"]["no_losses"] is True and r1["inputs"]["losses"] == 0
    assert "Infinity" not in json.dumps(ev)
    assert "inf (no losses)" in risk.describe(ev)
    assert risk.evaluate(make_run(base, "r1-even2", [
        bar_rec(None, 100_000.0, fill("buy", 10.0, 100.0)),
        bar_rec(None, 100_000.0, fill("sell", 10.0, 99.0))], equity=100_000.0)
    )["rules"]["r1_profit_factor"]["inputs"]["no_losses"] is False


def test_r1_reads_only_the_epoch_and_the_last_60():
    base = _fresh("r1c")
    # 60 losses then 60 wins: the last-60 window is all wins -> PF inf
    trades = losing_trades(60)
    for _ in range(60):
        trades.append(bar_rec(None, 100_000.0, fill("buy", 10.0, 100.0)))
        trades.append(bar_rec(None, 100_000.0, fill("sell", 10.0, 101.0)))
    run = make_run(base, "r1-window", trades, equity=100_000.0)
    r1 = risk.evaluate(run)["rules"]["r1_profit_factor"]
    assert r1["value"] is None and r1["inputs"]["no_losses"] and not r1["trip"]
    # an epoch that leaves the first 90 trades behind sees 30 wins: unarmed
    run.state["risk_epoch"] = {"bar": run.state["forward_start"], "equity": 100_000.0,
                               "closed_trades_before": 90}
    r1 = risk.evaluate(run)["rules"]["r1_profit_factor"]
    assert r1["inputs"]["closed_in_epoch"] == 30 and not r1["armed"], r1


# ────────────────────────────── R2 equity band ──────────────────────────────

def _declining(n: int, e0: float, e_end: float) -> list[dict]:
    path = np.geomspace(e0, e_end, n + 1)[1:]
    return [bar_rec(None, float(e)) for e in path]


def test_r2_unarmed_below_20_bars_and_trips_below_the_band():
    base = _fresh("r2")
    run = make_run(base, "r2-19", _declining(19, 100_000.0, 85_000.0), equity=85_000.0)
    ev = risk.evaluate(run)
    r2 = ev["rules"]["r2_equity_band"]
    assert r2["inputs"]["n"] == 19 and not r2["armed"] and not r2["trip"], r2
    assert abs(r2["value"] - np.log(0.85)) < 1e-6
    assert r2["inputs"]["sigma_bar"] > 0 and r2["inputs"]["drift"] == 0.0

    run = make_run(base, "r2-25", _declining(25, 100_000.0, 85_000.0), equity=85_000.0)
    ev = risk.evaluate(run)
    r2 = ev["rules"]["r2_equity_band"]
    sigma, n = r2["inputs"]["sigma_bar"], r2["inputs"]["n"]
    assert n == 25 and r2["armed"], r2
    assert abs(r2["threshold"] - (-risk.BAND_SIGMAS * sigma * np.sqrt(25))) < 1e-6   # both rounded to 8 dp
    assert r2["value"] < r2["threshold"] and r2["trip"], r2
    assert ev["tripped"] == "r2_equity_band"
    assert r2["inputs"]["E_0"] == 100_000.0 and r2["inputs"]["E"] == 85_000.0
    # a smooth decline has no realised vol to speak of: R3 armed, not tripped
    assert ev["rules"]["r3_realised_vol"]["armed"] and not ev["rules"]["r3_realised_vol"]["trip"]

    # inside the band: armed, not tripped
    run = make_run(base, "r2-ok", _declining(25, 100_000.0, 99_000.0), equity=99_000.0)
    r2 = risk.evaluate(run)["rules"]["r2_equity_band"]
    assert r2["armed"] and not r2["trip"], r2


def test_r2_sigma_comes_from_the_seed_through_the_engine():
    base = _fresh("r2b")
    run = make_run(base, "r2-seed", _declining(25, 100_000.0, 99_000.0), equity=99_000.0)
    seed = risk.seed_returns(run)
    bars = run.bars()
    fs = pd.Timestamp(run.state["forward_start"])
    # bars with index <= forward_start, less the vol lookback
    assert len(seed) == int((bars.index <= fs).sum()) - run.config["vol_lookback"], len(seed)
    assert seed.index[-1] <= fs
    r2 = risk.evaluate(run)["rules"]["r2_equity_band"]
    assert abs(r2["inputs"]["sigma_bar"] - float(seed.std(ddof=1))) < 1e-7
    # buy_hold at a 15% vol target on 1.2%-a-day noise sizes to ~0.8, so the
    # engine's per-bar sigma lands near 0.8 x 1.2%
    assert 0.006 < r2["inputs"]["sigma_bar"] < 0.014, r2["inputs"]["sigma_bar"]


def test_false_trip_rate_is_deterministic_and_sensible():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0, 0.01, 400))
    a = risk.false_trip_rate(r, paths=300)
    b = risk.false_trip_rate(r, paths=300)
    assert a == b and 0.0 <= a <= 1.0
    # a 2-sigma band checked at every n is crossed far more often than 2.3%
    assert 0.02 < a < 0.6, a
    assert risk.false_trip_rate(pd.Series([0.0] * 50)) is None
    assert risk.false_trip_rate(pd.Series(dtype=float)) is None
    # more drift down, more trips
    down = risk.false_trip_rate(r - 0.002, paths=300)
    assert down > a, (down, a)


# ────────────────────────────── R3 realised vol ──────────────────────────────

def _wild(n: int, e0: float = 100_000.0, swing: float = 0.05) -> list[dict]:
    out, e = [], e0
    for i in range(n):
        e = e * (1 + swing) if i % 2 == 0 else e / (1 + swing)
        out.append(bar_rec(None, e))
    return out


def test_r3_unarmed_below_20_bars_and_trips_above_2x_target():
    base = _fresh("r3")
    run = make_run(base, "r3-19", _wild(19), equity=100_000.0)
    r3 = risk.evaluate(run)["rules"]["r3_realised_vol"]
    assert r3["inputs"]["n"] == 19 and not r3["armed"] and not r3["trip"], r3
    assert r3["value"] is not None and r3["value"] > 0.3

    run = make_run(base, "r3-20", _wild(20), equity=100_000.0)
    ev = risk.evaluate(run)
    r3 = ev["rules"]["r3_realised_vol"]
    assert r3["armed"] and r3["trip"] and r3["threshold"] == 0.3, r3
    assert r3["inputs"]["bars_used"] == 20 and r3["inputs"]["ppy"] == 252.0
    assert ev["tripped"] == "r3_realised_vol"
    # equity is back where it started, so R2 is armed and not tripped
    assert ev["rules"]["r2_equity_band"]["armed"] and not ev["rules"]["r2_equity_band"]["trip"]

    # a quiet path: armed, not tripped
    run = make_run(base, "r3-quiet", _wild(30, swing=0.001), equity=100_000.0)
    r3 = risk.evaluate(run)["rules"]["r3_realised_vol"]
    assert r3["armed"] and not r3["trip"] and r3["value"] < 0.05, r3


def test_r3_uses_the_last_vol_lookback_bars_and_is_disabled_without_a_target():
    base = _fresh("r3b")
    # 80 quiet bars then 60 wild ones: the window is the last 60 -> trips
    recs = _wild(80, swing=0.001) + _wild(60, e0=100_000.0)
    run = make_run(base, "r3-window", recs, equity=100_000.0)
    r3 = risk.evaluate(run)["rules"]["r3_realised_vol"]
    assert r3["inputs"]["bars_used"] == 60 and r3["trip"], r3
    # 60 wild then 80 quiet: the window is quiet -> no trip
    recs = _wild(60) + _wild(80, swing=0.001)
    run = make_run(base, "r3-window2", recs, equity=100_000.0)
    r3 = risk.evaluate(run)["rules"]["r3_realised_vol"]
    assert r3["inputs"]["bars_used"] == 60 and not r3["trip"], r3
    # vol_target None: disabled, and said so
    run = make_run(base, "r3-none", _wild(30), equity=100_000.0, vol_target=None)
    r3 = risk.evaluate(run)["rules"]["r3_realised_vol"]
    assert not r3["armed"] and not r3["trip"] and r3["threshold"] is None
    assert "disabled" in r3.get("note", ""), r3


# ────────────────────────────── epoch, halt, resume ──────────────────────────────

def test_trip_writes_marker_journal_and_alert_with_the_same_numbers():
    base = _fresh("trip")
    _quiet_alerts()
    with Env(None, None):
        run = make_run(base, "trip-r2", _declining(25, 100_000.0, 85_000.0), equity=85_000.0)
        assert not run.is_halted
        halt = run._risk_check()
    assert halt and halt["rule"] == "r2_equity_band" and run.is_halted
    marker = run._halted_marker()
    assert set(marker) >= {"at", "rule", "value", "threshold", "inputs", "epoch", "unhalt"}
    assert marker["unhalt"] == risk.UNHALT and marker["value"] < marker["threshold"]
    recs = _records(run.root)
    halted_line = [r for r in recs if r["type"] == "halted"]
    risk_line = [r for r in recs if r["type"] == "risk"]
    assert len(halted_line) == 1 and len(risk_line) == 1
    for key in ("at", "rule", "value", "threshold", "inputs", "epoch", "unhalt"):
        assert halted_line[0][key] == marker[key], key
    assert risk_line[0]["tripped"] == "r2_equity_band"
    log = (base / alerts.ALERTS_FILE).read_text(encoding="utf-8")
    assert "[halt]" in log and "trip-r2" in log and "r2_equity_band" in log
    last = log.strip().splitlines()[-1]
    fields = json.loads(last[last.index(' {"') + 1:])
    assert fields["value"] == marker["value"] and fields["threshold"] == marker["threshold"]
    assert fields["inputs"] == marker["inputs"]
    assert run.state["risk"]["tripped"] == "r2_equity_band"
    # halted: the rules are not evaluated again
    assert run._risk_check() is None


def test_resume_starts_a_new_epoch_and_does_not_retrip():
    base = _fresh("resume")
    _quiet_alerts()
    with Env(None, None):
        run = make_run(base, "res-r2", _declining(25, 100_000.0, 85_000.0), equity=85_000.0)
        assert run._risk_check() is not None and run.is_halted
        marker = run._halted_marker()
        rec = run.resume("looked at it, the model is fine")
        assert not run.is_halted and rec["halt"] == marker
        ep = run.state["risk_epoch"]
        assert ep == {"bar": run.state["last_bar"], "equity": 85_000.0, "closed_trades_before": 0}
        assert run.state["risk"] is None
        resumed = [r for r in _records(run.root) if r["type"] == "resumed"]
        assert len(resumed) == 1 and resumed[0]["reason"].startswith("looked") \
            and resumed[0]["halt"] == marker
        # the same numbers, a new epoch: nothing armed, no re-trip
        ev = risk.evaluate(run)
        assert ev["epoch_bars"] == 0 and ev["armed"] == [] and ev["tripped"] is None
        assert run._risk_check() is None and not run.is_halted
        assert risk.epoch(run) == ep
    # the default epoch is the forward start and is not stored in state
    run2 = make_run(base, "default-epoch", _declining(5, 100_000.0, 99_000.0), equity=99_000.0)
    assert "risk_epoch" not in run2.state
    assert risk.epoch(run2) == {"bar": run2.state["forward_start"], "equity": 100_000.0,
                                "closed_trades_before": 0}


def test_manual_halt_refused_on_stopped_and_stop_wins():
    base = _fresh("stopped")
    run = make_run(base, "stp", _declining(5, 100_000.0, 99_000.0), equity=99_000.0)
    run.stop("done")
    try:
        run.halt("nope")
        assert False, "halt on a stopped run must be refused"
    except RuntimeError:
        pass
    assert not run.is_halted
    run2 = make_run(base, "hlt", _declining(5, 100_000.0, 99_000.0), equity=99_000.0)
    m = run2.halt("by hand")
    assert m["rule"] == "manual" and m["reason"] == "by hand" and run2.is_halted
    run2.stop("and then stopped")
    assert run2.is_stopped and run2.is_halted           # the marker stays as history
    assert run2.poll().get("stopped") is True             # STOPPED wins: not polled


def test_journal_risk_line_only_when_a_flag_changes():
    base = _fresh("flags")
    run = make_run(base, "flags", _declining(5, 100_000.0, 99_900.0), equity=99_900.0)
    n0 = len(_records(run.root))
    assert run._risk_check() is None
    assert len(_records(run.root)) == n0            # nothing armed: no line
    assert run.state["risk"]["armed"] == []
    assert run._risk_check() is None
    assert len(_records(run.root)) == n0
    ev1 = risk.evaluate(run)
    assert not risk.changed(run.state["risk"], ev1)
    ev1["rules"]["r2_equity_band"]["armed"] = True
    assert risk.changed(run.state["risk"], ev1)


# ────────────────────────────── halt honoured by poll ──────────────────────────────

def _poll_until_fill(run: P.PaperRun, limit: int = 20) -> dict:
    for _ in range(limit):
        res = run.poll()
        for ev in res.get("events", []):
            if ev.get("fill"):
                return ev
    raise AssertionError("no fill within the poll limit")


def test_halt_honoured_by_poll_on_a_replay_run():
    base = _fresh("replay-halt")
    _quiet_alerts()
    with Env(None, None):
        run = replay_run(base, "rh")
        first = run.poll()
        assert first.get("bootstrapped") and run.state["pending"], first
        _poll_until_fill(run)
        fills_before = run.state["fills"]
        # a decision pending from the last bar is written off by the halt (buy_hold
        # inside its band rarely leaves one, so it is planted by hand)
        run.poll()
        risk_before = run.state.get("risk")
        assert risk_before is not None                  # the rules ran on the live polls
        run.state["pending"] = {"bar": run.state["last_bar"], "target": 0.5, "signal": 1.0}
        run._save_state()
        marker = run.halt("manual test halt")
        assert run.is_halted and marker["rule"] == "manual"
        assert run.state["pending"] is None
        unfilled = [r for r in _records(run.root) if r["type"] == "unfilled"]
        assert unfilled and unfilled[-1]["unfilled"]["why_not"].startswith("halted")
        n_before = len(_records(run.root))

        for _ in range(3):
            res = run.poll()
            assert res.get("halted") is True and res["new_bars"] == 1, res
            ev = res["events"][0]
            assert ev["halted"] is True and ev["order"] is None and ev["fill"] is None
            assert ev["signal"] is None and ev["target"] is None
            assert run.state["pending"] is None
        recs = _records(run.root)
        bars = [r for r in recs[n_before:] if r["type"] == "bar"]
        assert len(bars) == 3 and all(r["halted"] is True and r["order"] is None for r in bars)
        assert all(r["fill"] is None for r in bars)
        assert run.state["fills"] == fills_before
        assert run.state["bars_forward"] >= 3            # still journaling and marking
        assert run.state.get("risk") == risk_before      # rules not re-evaluated while halted
        assert not [r for r in recs[n_before:] if r["type"] == "risk"]

        # resume: the next poll decides again and the record carries no `halted`
        run.resume("done looking")
        res = run.poll()
        assert res.get("halted") is None
        ev = res["events"][0]
        assert "halted" not in ev and ev["signal"] == 1.0 and ev["target"] is not None
        bar_keys = set(ev) - {"halt"}
        assert "risk" not in bar_keys


def test_trip_inside_a_poll_clears_the_order_and_the_dry_run_writes_nothing():
    base = _fresh("poll-trip")
    _quiet_alerts()
    with Env(None, None):
        run = replay_run(base, "pt")
        run.poll()
        _poll_until_fill(run)
        for _ in range(3):
            run.poll()
        # a trip on the next poll, whatever the numbers say: the rules are
        # answered by a stand-in so the poll's own handling can be checked
        real = P.risk_mod.evaluate

        def tripping(run_, now=None, seed=None):
            ev = real(run_, now=now, seed=seed)
            r3 = ev["rules"]["r3_realised_vol"]
            r3.update(armed=True, trip=True, value=0.9, threshold=0.3)
            ev["tripped"], ev["armed"] = "r3_realised_vol", ["r3_realised_vol"]
            return ev

        P.risk_mod.evaluate = tripping
        try:
            before = {p.name: p.read_bytes() for p in run.root.iterdir()}
            dry = run.poll(dry_run=True)
            assert dry.get("dry_run") and not run.is_halted
            ev = dry["events"][0]
            assert ev.get("halt", {}).get("would_halt") is True
            assert {p.name: p.read_bytes() for p in run.root.iterdir()} == before
            assert not (base / alerts.ALERTS_FILE).exists()

            res = run.poll()
            assert run.is_halted and res.get("halted") is True
            ev = res["events"][0]
            assert "halt" in ev and ev["halt"]["rule"] == "r3_realised_vol"
            assert "halted" not in ev                        # the trip poll's record is a normal one
            assert run.state["pending"] is None
            recs = _records(run.root)
            assert recs[-1]["type"] in ("unfilled", "halted")
            halted = [r for r in recs if r["type"] == "halted"]
            assert len(halted) == 1 and halted[0]["value"] == 0.9
            assert run._halted_marker()["inputs"] == halted[0]["inputs"]
            if ev.get("order"):
                unf = [r for r in recs if r["type"] == "unfilled"]
                assert unf and unf[-1]["unfilled"]["why_not"].startswith("halted")
            log = (base / alerts.ALERTS_FILE).read_text(encoding="utf-8")
            assert "[halt]" in log and "r3_realised_vol" in log
        finally:
            P.risk_mod.evaluate = real
        # the next poll: a bar with halted: true, no order, no fill
        res = run.poll()
        ev = res["events"][0]
        assert ev["halted"] is True and ev["order"] is None and ev["fill"] is None


def test_live_poll_bar_record_has_no_new_keys():
    base = _fresh("keys")
    _quiet_alerts()
    with Env(None, None):
        run = replay_run(base, "keys")
        run.poll()
        for _ in range(5):
            res = run.poll()
        rec = [r for r in _records(run.root) if r["type"] == "bar"][-1]
        assert set(rec) == {"type", "at", "bar", "open", "close", "fill", "equity", "held",
                            "drawdown", "signal", "target", "vol_scale", "order"} | (
            {"unfilled"} if "unfilled" in rec else set()), sorted(rec)
        st = json.loads(run.state_path.read_text(encoding="utf-8"))
        assert "risk" in st and "risk_epoch" not in st


# ────────────────────────────── heartbeat ──────────────────────────────

def _hb_base(name: str, now: datetime) -> Path:
    base = _fresh(name)
    for run_id, age, stopped in (("fresh", 60, False), ("stale", 3600, False),
                                 ("stopped", 3600, True)):
        root = base / run_id
        root.mkdir()
        cfg = {"run_id": run_id, "strategy": "buy_hold", "feed": {"kind": "csv", "path": "x"}}
        (root / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        (root / "state.json").write_text(json.dumps({"updated": _iso(now - timedelta(seconds=age))}),
                                         encoding="utf-8")
        if stopped:
            (root / "STOPPED").write_text("{}", encoding="utf-8")
    (base / "book.json").write_text(json.dumps({"as_of": _iso(now - timedelta(seconds=30)),
                                                "tick_s": 300}), encoding="utf-8")
    return base


def _touch(base: Path, run_id: str, now: datetime) -> None:
    (base / run_id / "state.json").write_text(json.dumps({"updated": _iso(now)}), encoding="utf-8")


def _book(base: Path, now: datetime) -> None:
    """The tick keeps writing the book while the clock moves; without this the
    book would go stale on its own and muddy the count."""
    (base / "book.json").write_text(json.dumps({"as_of": _iso(now - timedelta(seconds=30)),
                                                "tick_s": 300}), encoding="utf-8")


def test_heartbeat_alerts_once_repeats_after_6h_and_logs_recovery():
    t0 = NOW
    base = _hb_base("hb", t0)
    sent = []

    def notify(base_dir, kind, message, fields=None, push=None):
        sent.append({"kind": kind, "message": message, "push": push})
        return {"kind": kind}

    rep = risk.heartbeat(base, now=t0, notify=notify)
    assert [a["item"] for a in rep["alerts"]] == ["stale"], rep["alerts"]
    assert len(sent) == 1 and sent[0]["kind"] == "stale" and "600 s" in sent[0]["message"]
    by = {it["item"]: it for it in rep["items"]}
    assert by["fresh"]["action"] == "ok" and not by["fresh"]["stale"]
    assert by["stopped"]["skipped"] == "stopped"
    assert by["book.json"]["action"] == "ok" and by["book.json"]["limit_s"] == 600
    doc = json.loads((base / risk.HEARTBEAT_FILE).read_text(encoding="utf-8"))
    assert set(doc["items"]) == {"stale"} and doc["items"]["stale"]["last_alert"] == _iso(t0)

    # five minutes on: still stale, not repeated
    _book(base, t0 + timedelta(minutes=5))
    rep = risk.heartbeat(base, now=t0 + timedelta(minutes=5), notify=notify)
    assert rep["alerts"] == [] and len(sent) == 1
    assert {it["item"]: it.get("action") for it in rep["items"]}["stale"] == "suppressed"
    # 5h59 on: still not
    _book(base, t0 + timedelta(hours=5, minutes=59))
    _touch(base, "fresh", t0 + timedelta(hours=5, minutes=58))
    rep = risk.heartbeat(base, now=t0 + timedelta(hours=5, minutes=59), notify=notify)
    assert len(sent) == 1
    # six hours on: once more
    t6 = t0 + timedelta(hours=6)
    _book(base, t6)
    _touch(base, "fresh", t6 - timedelta(seconds=60))
    rep = risk.heartbeat(base, now=t6, notify=notify)
    assert len(sent) == 2 and sent[1]["kind"] == "stale"
    doc = json.loads((base / risk.HEARTBEAT_FILE).read_text(encoding="utf-8"))
    assert doc["items"]["stale"]["last_alert"] == _iso(t6)
    assert doc["items"]["stale"]["stale_since"] == _iso(t0)

    # it comes back: recovery is logged, not pushed, and the state clears
    t7 = t6 + timedelta(minutes=10)
    _book(base, t7)
    _touch(base, "fresh", t7 - timedelta(seconds=60))
    _touch(base, "stale", t7 - timedelta(seconds=30))
    rep = risk.heartbeat(base, now=t7, notify=notify)
    assert len(sent) == 3 and sent[2]["kind"] == "recovered" and sent[2]["push"] is False
    doc = json.loads((base / risk.HEARTBEAT_FILE).read_text(encoding="utf-8"))
    assert doc["items"] == {}
    # --push-recovery turns the push on
    (base / "stale" / "state.json").write_text(json.dumps({"updated": _iso(t0)}), encoding="utf-8")
    risk.heartbeat(base, now=t7, notify=notify)
    _touch(base, "stale", t7)
    _book(base, t7 + timedelta(minutes=5))
    _touch(base, "fresh", t7 + timedelta(minutes=4))
    risk.heartbeat(base, now=t7 + timedelta(minutes=5), notify=notify, push_recovery=True)
    assert sent[-1]["kind"] == "recovered" and sent[-1]["push"] is True


def test_heartbeat_book_stale_and_missing_and_dry_run_writes_nothing():
    t0 = NOW
    base = _hb_base("hb2", t0)
    sent = []

    def notify(base_dir, kind, message, fields=None, push=None):
        sent.append(kind)
        return {}

    rep = risk.heartbeat(base, now=t0, dry_run=True, notify=notify)
    assert rep["dry_run"] and [a["item"] for a in rep["alerts"]] == ["stale"]
    assert sent == [] and not (base / risk.HEARTBEAT_FILE).exists()
    # book.json 20 minutes old: stale; missing: stale
    (base / "book.json").write_text(json.dumps({"as_of": _iso(t0 - timedelta(minutes=20)),
                                                "tick_s": 300}), encoding="utf-8")
    rep = risk.heartbeat(base, now=t0, notify=notify)
    assert sorted(a["item"] for a in rep["alerts"]) == ["book.json", "stale"]
    (base / "book.json").unlink()
    rep = risk.heartbeat(base, now=t0 + timedelta(minutes=5), notify=notify)
    by = {it["item"]: it for it in rep["items"]}
    assert by["book.json"]["stale"] and by["book.json"]["age_s"] is None
    assert rep["alerts"] == []                              # both already alerted
    # tick_s from the run's execution block widens its limit
    cfg = json.loads((base / "stale" / "config.json").read_text(encoding="utf-8"))
    cfg["execution"] = {"tick_s": 3600}
    (base / "stale" / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    rep = risk.heartbeat(base, now=t0 + timedelta(minutes=10), notify=notify)
    by = {it["item"]: it for it in rep["items"]}
    assert by["stale"]["limit_s"] == 7200 and not by["stale"]["stale"]
    assert by["stale"]["action"] == "recovered"
    # the real notify path with Telegram unset: the log line is written, nothing else
    base2 = _hb_base("hb3", t0)
    rec = _quiet_alerts()
    with Env(None, None):
        risk.heartbeat(base2, now=t0)
    log = (base2 / alerts.ALERTS_FILE).read_text(encoding="utf-8")
    assert log.count("ALERT ") == 1 and "[stale] stale is stale" in log and rec.calls == []


# ────────────────────────────── alerts ──────────────────────────────

def test_alert_always_logs_and_book_alert_delegates():
    base = _fresh("alerts")
    rec = Recorder()
    alerts._post = rec
    with Env(None, None):
        out = alerts.notify(base, "book", "the book is stale")
        assert out["logged"] and not out["pushed"]
        book.alert(base, "k not applied")
    lines = (base / alerts.ALERTS_FILE).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        head, _, rest = line.partition(" [")
        assert head.startswith("ALERT 20") and rest.startswith("book] ")
        datetime.fromisoformat(head.split(" ")[1])
    assert lines[1].endswith("[book] k not applied")
    assert rec.calls == []


def test_alert_unset_env_never_touches_the_transport():
    base = _fresh("alerts-unset")
    rec = Recorder(fail=AssertionError("must not be called"))
    alerts._post = rec
    with Env(None, None):
        assert not alerts.configured()
        out = alerts.notify(base, "halt", "x")
        assert out["logged"] and not out["pushed"] and "not set" in out["why_not_pushed"]
    with Env("only-token", None):
        assert not alerts.configured()
        assert not alerts.notify(base, "fill", "y")["pushed"]
    assert rec.calls == []
    assert (base / alerts.ALERTS_FILE).read_text(encoding="utf-8").count("ALERT ") == 2


def test_alert_set_env_posts_the_right_payload_and_keeps_the_token_out_of_the_log():
    base = _fresh("alerts-set")
    rec = Recorder()
    alerts._post = rec
    token, chat = "123456:ABC-fake-token-xyz-0123456789", "987654321"
    with Env(token, chat):
        assert alerts.configured()
        out = alerts.notify(base, "halt", "run HALTED by r1", fields={"rule": "r1", "value": 0.5})
    assert out["pushed"] and out["telegram"]["sent"] and out["telegram"]["attempts"] == 1
    assert len(rec.calls) == 1
    call = rec.calls[0]
    assert call["url"] == f"https://api.telegram.org/bot{token}/sendMessage"
    assert call["timeout"] == 10.0
    assert call["body"]["chat_id"] == chat
    assert call["body"]["disable_web_page_preview"] is True
    assert call["body"]["text"].startswith("run HALTED by r1") and "rule: r1" in call["body"]["text"]
    log = (base / alerts.ALERTS_FILE).read_text(encoding="utf-8")
    assert token not in log and chat not in log
    assert log.count("ALERT ") == 1 and "[halt] run HALTED by r1" in log
    assert token not in json.dumps(out)
    # log-only kinds are not pushed even when configured
    with Env(token, chat):
        assert not alerts.notify(base, "book", "z")["pushed"]
    assert len(rec.calls) == 1


def test_alert_transport_failure_is_logged_retried_once_and_never_raises():
    base = _fresh("alerts-fail")
    token = "99999:fake-token-fake-token-fake"
    rec = Recorder(fail=OSError(f"boom https://api.telegram.org/bot{token}/sendMessage"))
    alerts._post = rec
    with Env(token, "1"):
        out = alerts.notify(base, "stale", "x is stale")
    assert out["logged"] and not out["pushed"] and out["telegram"]["attempts"] == 2
    assert len(rec.calls) == 2
    log = (base / alerts.ALERTS_FILE).read_text(encoding="utf-8")
    lines = log.splitlines()
    assert len(lines) == 2 and "[stale] x is stale" in lines[0]
    assert "[telegram] push of [stale] failed after 2 attempt(s)" in lines[1]
    assert token not in log and "<token>" in lines[1]
    # a non-2xx status is a failure too
    rec = Recorder(status=500)
    alerts._post = rec
    with Env(token, "1"):
        out = alerts.notify(base, "fill", "y")
    assert not out["pushed"] and len(rec.calls) == 2


# ────────────────────────────── review fixes ──────────────────────────────
# The adversarial reviews of the first build: R2's sigma diluted by the
# strategy's warmup, R3 annualised off a window shorter than a session, the dry
# run mixing disk and memory, the heartbeat's timing and its failed pushes, and
# the transport's unbounded wall clock.

def _engine_returns(run: P.PaperRun):
    from quantlab import engine
    cfg = run.config
    bars = run.bars()
    seed = bars[bars.index <= pd.Timestamp(run.state["forward_start"])]
    bps = cfg["cost_bps"] / 2
    return engine.run(seed, P.REGISTRY[cfg["strategy"]].signal(seed, **cfg["params"]),
                      engine.CostModel(commission_bps=bps, slippage_bps=bps),
                      vol_target=cfg["vol_target"], vol_lookback=cfg["vol_lookback"],
                      max_leverage=cfg["max_leverage"], rebalance_band=cfg["rebalance_band"])


def test_r2_seed_starts_at_the_strategys_first_position():
    base = _fresh("r2-warmup")
    # tsmom holds nothing for its first 252 bars: on a 400-bar seed the old
    # sigma was the sd of ~190 exact zeros and ~150 real returns
    run = make_run(base, "warm", _declining(25, 100_000.0, 99_000.0), equity=99_000.0,
                   n_seed=400, strategy="tsmom")
    prof = risk.seed_profile(run)
    res = _engine_returns(run)
    pos = res.position.fillna(0.0).to_numpy()
    first = int(np.flatnonzero(pos != 0)[0])
    assert first > 200, first
    seed = prof["returns"]
    assert len(seed) == 400 - first and prof["warmup_bars"] == first - 60, prof
    assert prof["first_position_bar"] == str(run.bars().index[first]) and prof["note"] is None
    assert seed.iloc[0] != 0.0 and not (seed == 0.0).all()
    diluted = float(res.returns.iloc[60:].std(ddof=1))
    sigma = float(seed.std(ddof=1))
    assert sigma > diluted * 1.2, (sigma, diluted)         # the zeros were shrinking it
    ev = risk.evaluate(run)
    r2 = ev["rules"]["r2_equity_band"]
    assert r2["armed"] and abs(r2["inputs"]["sigma_bar"] - sigma) < 1e-7
    assert r2["inputs"]["warmup_bars"] == first - 60 and r2["inputs"]["seed_bars"] == len(seed)
    assert r2["inputs"]["seed_history_bars"] == 400 and r2["inputs"]["min_seed_bars"] == 60
    # the same series through evaluate(seed=<bare series>) and seed_returns
    assert risk.seed_returns(run).equals(seed)
    assert risk.evaluate(run, seed=seed)["rules"]["r2_equity_band"]["inputs"]["sigma_bar"] \
        == r2["inputs"]["sigma_bar"]
    # buy_hold has no warmup beyond the vol lookback: nothing changes for it
    run = make_run(base, "bh", _declining(25, 100_000.0, 99_000.0), equity=99_000.0)
    prof = risk.seed_profile(run)
    assert prof["warmup_bars"] == 0 and len(prof["returns"]) == 300 - 60


def test_r2_is_disabled_when_the_seed_barely_covers_the_strategy():
    base = _fresh("r2-short")
    # 300 seed bars, 252 of them warmup: fewer than vol_lookback survive
    run = make_run(base, "short", _declining(25, 100_000.0, 99_000.0), equity=99_000.0,
                   n_seed=300, strategy="tsmom")
    prof = risk.seed_profile(run)
    assert prof["returns"].empty and "warmup" in prof["note"] and "need 60" in prof["note"], prof
    ev = risk.evaluate(run)
    r2 = ev["rules"]["r2_equity_band"]
    assert not r2["armed"] and not r2["trip"] and r2["inputs"]["n"] == 25
    assert "rule disabled" in r2["note"] and r2["inputs"]["sigma_bar"] is None, r2
    assert r2["inputs"]["seed_bars"] == 0 and r2["inputs"]["min_seed_bars"] == 60
    assert risk.false_trip_rate(prof["returns"]) is None
    # the other two rules are untouched by it
    assert ev["rules"]["r3_realised_vol"]["armed"]
    # a seed the strategy never invests in at all
    run = make_run(base, "never", _declining(25, 100_000.0, 99_000.0), equity=99_000.0,
                   n_seed=200, strategy="tsmom")
    prof = risk.seed_profile(run)
    assert prof["returns"].empty and "never holds a position" in prof["note"], prof
    assert not risk.evaluate(run)["rules"]["r2_equity_band"]["armed"]


def _intraday_bars(days: int, per_day: int = 78, seed: int = 5) -> pd.DataFrame:
    """5-minute bars, a 6.5 h session a day on business days."""
    days_idx = pd.bdate_range("2026-03-02", periods=days)
    idx = pd.DatetimeIndex([d + pd.Timedelta(minutes=5 * i) + pd.Timedelta(hours=9, minutes=30)
                            for d in days_idx for i in range(per_day)])
    n = len(idx)
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    noise = np.abs(rng.normal(0, 0.0005, n)) * close
    return pd.DataFrame({"open": close, "high": close + noise, "low": close - noise,
                         "close": close, "volume": np.full(n, 1e5)}, index=idx)


def test_r3_annualises_with_the_run_bar_store_not_the_epoch_window():
    base = _fresh("r3-ppy")
    bars = _intraday_bars(days=10)                    # 780 five-minute bars
    fwd = _wild(40, swing=0.001)
    run = make_run(base, "intra", fwd, equity=100_000.0, bars=bars)
    r3 = risk.evaluate(run)["rules"]["r3_realised_vol"]
    full = float(data.periods_per_year(run.bars().index))
    window = pd.DatetimeIndex(pd.to_datetime([r["bar"] for r in fwd]))
    assert abs(full - 252 * 78) < 1e-6, full
    assert r3["inputs"]["ppy"] == full and r3["inputs"]["ppy_from_bars"] == len(bars), r3
    # the window alone (40 bars, under a session) would read 252 x 40
    assert float(data.periods_per_year(window)) < full / 1.5
    assert abs(r3["value"] - r3["inputs"]["sd_per_bar"] * np.sqrt(full)) < 1e-6   # 8-dp rounding
    assert r3["armed"]
    # the same equity path on daily bars annualises at 252 and reads ~9x lower
    run_d = make_run(base, "daily", _wild(40, swing=0.001), equity=100_000.0)
    r3_d = risk.evaluate(run_d)["rules"]["r3_realised_vol"]
    assert r3_d["inputs"]["ppy"] == 252.0
    assert abs(r3["value"] / r3_d["value"] - np.sqrt(78)) < 0.5, (r3["value"], r3_d["value"])
    # a 7-day calendar reads 365.25 from the store even with 25 epoch bars,
    # where the window alone (fewer than 28 gaps) would have said 252
    idx = pd.date_range("2025-01-01", periods=325, freq="D")
    daily = data.synthetic(n=325, seed=2)
    daily.index = idx
    run_7 = make_run(base, "seven", _wild(25, swing=0.001), equity=100_000.0, bars=daily)
    r3_7 = risk.evaluate(run_7)["rules"]["r3_realised_vol"]
    assert r3_7["inputs"]["ppy"] == 365.25 and r3_7["inputs"]["bars_used"] == 25, r3_7


def test_dry_run_reads_the_rules_on_the_records_it_would_have_journaled():
    base = _fresh("dry-consistent")
    _quiet_alerts()
    # a csv feed, not the replay feed: the replay's cursor advances on every
    # fetch, dry run included, so its real poll would see the NEXT bar. A file
    # something appends to hands the same bars to the dry run and the real poll.
    bars = data.synthetic(n=260, seed=3)
    csv = base / "dc-bars.csv"
    _write_bars(csv, bars.iloc[:250])
    with Env(None, None):
        run = P.PaperRun.create(base, "dc", "buy_hold", {"kind": "csv", "path": str(csv)},
                                min_history=100, vol_lookback=60)
        run.poll()                                    # bootstrap on 250 bars
        for k in range(251, 257):
            _write_bars(csv, bars.iloc[:k])
            run.poll()
        _write_bars(csv, bars.iloc[:257])
        on_disk = len(_records(run.root))
        before = {p.name: p.read_bytes() for p in run.root.iterdir()}
        dry = run.poll(dry_run=True)
        assert dry.get("dry_run") and dry["new_bars"] == 1
        assert {p.name: p.read_bytes() for p in run.root.iterdir()} == before
        dry_ev = dry["risk"]
        # the dry run counted the bar it did not journal, and the buffer is gone after
        assert dry_ev["epoch_bars"] == 7, dry_ev["epoch_bars"]
        assert run._dry_records == [] and len(run._records()) == on_disk
        # the dry run leaves its in-memory state behind (a CLI poll is a fresh
        # process): reload from disk, as `paper.py poll` does, then poll for real
        run = P.PaperRun.load(base, "dc")
        real = run.poll()
        real_ev = run.state["risk"]
        assert real_ev["epoch_bars"] == 7 == dry_ev["epoch_bars"]
        for name in risk.RULES:
            d, r = dry_ev["rules"][name], real_ev["rules"][name]
            assert d["armed"] == r["armed"] and d["trip"] == r["trip"], name
            assert d["value"] == r["value"] and d["threshold"] == r["threshold"], name
        assert dry_ev["rules"]["r2_equity_band"]["inputs"]["n"] == \
            real_ev["rules"]["r2_equity_band"]["inputs"]["n"] == 7
        assert dry_ev["rules"]["r2_equity_band"]["inputs"]["E"] == \
            real_ev["rules"]["r2_equity_band"]["inputs"]["E"]
        assert dry_ev["rules"]["r3_realised_vol"]["inputs"]["bars_used"] == 7
        # the real poll's bar (and any risk line) is on disk; the dry run's never was
        assert len(_records(run.root)) >= on_disk + 1 and real["new_bars"] == 1


def test_heartbeat_failed_push_stays_due_and_state_is_written_before_the_push():
    t0 = NOW
    base = _hb_base("hb-fail", t0)
    calls = []
    outcome = {"pushed": False, "telegram": {"error": "TimeoutError: no response within 10.0 s"}}

    def notify(base_dir, kind, message, fields=None, push=None):
        # by the time the transport is asked, the transition is on disk
        doc = json.loads((Path(base_dir) / risk.HEARTBEAT_FILE).read_text(encoding="utf-8"))
        calls.append({"kind": kind, "message": message, "on_disk": doc["items"]})
        return dict(outcome)

    rep = risk.heartbeat(base, now=t0, notify=notify)
    assert len(calls) == 1 and calls[0]["kind"] == "stale"
    assert calls[0]["on_disk"]["stale"] == {"stale_since": _iso(t0), "last_alert": None}
    assert rep["push"]["delivered"] is False
    doc = json.loads((base / risk.HEARTBEAT_FILE).read_text(encoding="utf-8"))
    assert doc["items"]["stale"]["last_alert"] is None
    assert doc["items"]["stale"]["last_attempt"] == _iso(t0)
    assert "TimeoutError" in doc["items"]["stale"]["last_error"]
    assert doc["items"]["stale"]["retry_not_before"] == _iso(t0 + timedelta(minutes=15))
    assert "not retried before" in risk.format_heartbeat(rep)
    # five and ten minutes on, transport still down: due, but in the backoff —
    # no attempt, and the report says why
    for minutes in (5, 10):
        t = t0 + timedelta(minutes=minutes)
        _book(base, t)
        _touch(base, "fresh", t - timedelta(seconds=60))
        rep = risk.heartbeat(base, now=t, notify=notify)
        assert len(calls) == 1 and rep["push"] is None, minutes
        by = {it["item"]: it for it in rep["items"]}
        assert by["stale"]["action"] == "backoff", by["stale"]
        assert by["stale"]["retry_not_before"] == _iso(t0 + timedelta(minutes=15))
        text = risk.format_heartbeat(rep)
        assert "backoff" in text and "TimeoutError" in text and "retry not before" in text
        doc = json.loads((base / risk.HEARTBEAT_FILE).read_text(encoding="utf-8"))
        assert doc["items"]["stale"]["last_attempt"] == _iso(t0)      # no new attempt
    # fifteen minutes on: tried again, still failing, a new backoff from now
    t1 = t0 + timedelta(minutes=15)
    _book(base, t1)
    _touch(base, "fresh", t1 - timedelta(seconds=60))
    rep = risk.heartbeat(base, now=t1, notify=notify)
    assert len(calls) == 2 and rep["push"]["delivered"] is False
    by = {it["item"]: it for it in rep["items"]}
    assert by["stale"]["action"] == "alert" and by["stale"]["delivered"] is False
    assert "still due" in risk.format_heartbeat(rep)
    doc = json.loads((base / risk.HEARTBEAT_FILE).read_text(encoding="utf-8"))
    assert doc["items"]["stale"]["last_alert"] is None
    assert doc["items"]["stale"]["last_attempt"] == _iso(t1)
    assert doc["items"]["stale"]["retry_not_before"] == _iso(t1 + timedelta(minutes=15))
    # the network is back at t+20 — but the backoff runs to t+30, so nothing
    # is tried at t+20 or t+25; at t+30 it is delivered, stamped, and the six
    # hours start from THAT pass
    outcome.update(pushed=True, telegram={"sent": True})
    for minutes in (20, 25):
        t = t0 + timedelta(minutes=minutes)
        _book(base, t)
        _touch(base, "fresh", t - timedelta(seconds=60))
        rep = risk.heartbeat(base, now=t, notify=notify)
        assert len(calls) == 2 and rep["push"] is None, minutes
    t2 = t0 + timedelta(minutes=30)
    _book(base, t2)
    _touch(base, "fresh", t2 - timedelta(seconds=60))
    rep = risk.heartbeat(base, now=t2, notify=notify)
    assert len(calls) == 3 and rep["push"]["delivered"] is True
    doc = json.loads((base / risk.HEARTBEAT_FILE).read_text(encoding="utf-8"))
    assert doc["items"]["stale"]["last_alert"] == _iso(t2)
    assert doc["items"]["stale"]["stale_since"] == _iso(t0)
    assert "last_error" not in doc["items"]["stale"]
    assert "retry_not_before" not in doc["items"]["stale"]
    # suppressed until t2 + 6 h, then once more
    for delta in (timedelta(minutes=5), timedelta(hours=5, minutes=59)):
        t = t2 + delta
        _book(base, t)
        _touch(base, "fresh", t - timedelta(seconds=60))
        rep = risk.heartbeat(base, now=t, notify=notify)
        assert len(calls) == 3 and rep["push"] is None, delta
        assert {it["item"]: it.get("action") for it in rep["items"]}["stale"] == "suppressed"
    t3 = t2 + timedelta(hours=6)
    _book(base, t3)
    _touch(base, "fresh", t3 - timedelta(seconds=60))
    rep = risk.heartbeat(base, now=t3, notify=notify)
    assert len(calls) == 4 and rep["push"]["delivered"] is True
    doc = json.loads((base / risk.HEARTBEAT_FILE).read_text(encoding="utf-8"))
    assert doc["items"]["stale"]["last_alert"] == _iso(t3)
    # Telegram not configured: log-only counts as delivered (nothing to retry)
    base2 = _hb_base("hb-noconf", t0)
    calls.clear()
    outcome.clear()
    outcome.update(pushed=False, why_not_pushed="TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set")
    rep = risk.heartbeat(base2, now=t0, notify=notify)
    assert rep["push"]["delivered"] is True
    doc = json.loads((base2 / risk.HEARTBEAT_FILE).read_text(encoding="utf-8"))
    assert doc["items"]["stale"]["last_alert"] == _iso(t0)
    _book(base2, t0 + timedelta(minutes=5))
    rep = risk.heartbeat(base2, now=t0 + timedelta(minutes=5), notify=notify)
    assert len(calls) == 1 and rep["push"] is None


def test_heartbeat_pushes_one_message_for_every_stale_item():
    t0 = NOW
    base = _hb_base("hb-one", t0)
    (base / "book.json").write_text(json.dumps({"as_of": _iso(t0 - timedelta(minutes=20)),
                                                "tick_s": 300}), encoding="utf-8")
    sent = []

    def notify(base_dir, kind, message, fields=None, push=None):
        sent.append({"kind": kind, "message": message, "fields": fields})
        return {"pushed": True}

    rep = risk.heartbeat(base, now=t0, notify=notify)
    assert len(sent) == 1 and sent[0]["kind"] == "stale"
    assert sorted(rep["push"]["items"]) == ["book.json", "stale"]
    assert "2 things are stale" in sent[0]["message"]
    assert "stale is stale" in sent[0]["message"] and "book.json is stale" in sent[0]["message"]
    assert sent[0]["fields"]["count"] == 2
    assert sorted(a["item"] for a in rep["alerts"]) == ["book.json", "stale"]
    doc = json.loads((base / risk.HEARTBEAT_FILE).read_text(encoding="utf-8"))
    assert all(v["last_alert"] == _iso(t0) for v in doc["items"].values())
    assert set(doc["items"]) == {"book.json", "stale"}
    text = risk.format_heartbeat(rep)
    assert "one message for 2 item(s)" in text
    # a single stale item keeps the one-line form the log has always had
    base2 = _hb_base("hb-one2", t0)
    sent.clear()
    risk.heartbeat(base2, now=t0, notify=notify)
    assert len(sent) == 1 and sent[0]["message"].startswith("stale is stale — state.updated")
    assert sent[0]["fields"] == {"item": "stale", "age_s": 3600, "limit_s": 600}


def test_alert_never_runs_past_its_deadline_when_the_transport_hangs():
    import time

    base = _fresh("alerts-hang")
    calls = []

    def hanging(url, body, timeout):
        calls.append(timeout)
        time.sleep(2.0)                               # DNS, a dripping server, anything
        return 200, json.dumps({"ok": True})

    saved = (alerts.TIMEOUT_S, alerts.DEADLINE_S, alerts.RETRY_MIN_S)
    alerts._post = hanging
    try:
        # two attempts, each abandoned at its budget, inside the deadline
        alerts.TIMEOUT_S, alerts.DEADLINE_S, alerts.RETRY_MIN_S = 0.2, 0.5, 0.05
        t = time.monotonic()
        with Env("11111:fake-fake-fake-fake-fake", "1"):
            out = alerts.notify(base, "halt", "x HALTED")
        took = time.monotonic() - t
        assert took < 1.0, took
        assert not out["pushed"] and out["telegram"]["attempts"] == 2
        assert "abandoned" in out["telegram"]["error"] and len(calls) == 2
        assert calls == [0.2, 0.2]                    # the socket timeout never exceeds the budget
        log = (base / alerts.ALERTS_FILE).read_text(encoding="utf-8")
        assert "failed after 2 attempt(s) in 0." in log, log
        # the first attempt eats the deadline: no retry
        calls.clear()
        alerts.TIMEOUT_S, alerts.DEADLINE_S, alerts.RETRY_MIN_S = 0.2, 0.25, 0.1
        with Env("11111:fake-fake-fake-fake-fake", "1"):
            out = alerts.notify(base, "fill", "y")
        assert out["telegram"]["attempts"] == 1 and len(calls) == 1
        assert "budget is spent" in out["telegram"]["error"], out
        # a transport that raises inside the worker is reported as before
        alerts.TIMEOUT_S, alerts.DEADLINE_S, alerts.RETRY_MIN_S = saved
        alerts._post = Recorder(fail=OSError("boom"))
        with Env("11111:fake-fake-fake-fake-fake", "1"):
            out = alerts.notify(base, "fill", "z")
        assert out["telegram"]["attempts"] == 2 and "OSError: boom" in out["telegram"]["error"]
    finally:
        alerts.TIMEOUT_S, alerts.DEADLINE_S, alerts.RETRY_MIN_S = saved


# ────────────────────────────── the second review's fixes ──────────────────────────────
# One line per notify in alerts.log, the heartbeat's retry backoff (in the
# failed-push test above), a resume made by another process while a poll holds
# the run, the rules failing closed, a journal line cut short by a crash, and
# what a halt writes off when the inflight never went out.

def test_alert_log_line_is_one_line_and_the_push_keeps_the_newlines():
    base = _fresh("alerts-flat")
    rec = Recorder()
    alerts._post = rec
    with Env("11111:fake-fake-fake-fake-fake", "1"):
        out = alerts.notify(base, "stale", "2 things are stale:\n  a is stale\n  b is stale",
                            fields={"count": 2})
    assert out["pushed"] and len(rec.calls) == 1
    assert rec.calls[0]["body"]["text"].startswith("2 things are stale:\n  a is stale\n  b is stale")
    lines = (base / alerts.ALERTS_FILE).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1, lines
    head, _, rest = lines[0].partition(" [")
    assert head.startswith("ALERT 20") and rest == \
        'stale] 2 things are stale: | a is stale | b is stale {"count": 2}'
    # the heartbeat's own multi-item message, through the real notify
    t0 = NOW
    base2 = _hb_base("hb-flat", t0)
    (base2 / "book.json").write_text(json.dumps({"as_of": _iso(t0 - timedelta(minutes=20)),
                                                 "tick_s": 300}), encoding="utf-8")
    rec = Recorder()
    alerts._post = rec
    with Env("11111:fake-fake-fake-fake-fake", "1"):
        rep = risk.heartbeat(base2, now=t0)
    assert sorted(rep["push"]["items"]) == ["book.json", "stale"] and rep["push"]["delivered"]
    assert "\n" in rec.calls[0]["body"]["text"]
    lines = (base2 / alerts.ALERTS_FILE).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and lines[0].count("ALERT ") == 1
    assert "[stale] 2 things are stale: | " in lines[0] and " | book.json is stale" in lines[0]


def test_alert_failed_push_line_is_one_line_whatever_the_transport_said():
    """A 502 page from Telegram's edge is several lines of HTML; the
    `[telegram] push ... failed` line that records it must still be one
    physical line, with the token masked — and so must heartbeat's last_error."""
    token = "11111:fake-fake-fake-fake-fake"
    html = ("<html>\n<head><title>502 Bad Gateway</title></head>\n<body>\n"
            f"bad https://api.telegram.org/bot{token}/sendMessage\n</body></html>\n")

    def non_2xx(url, body, timeout):
        return 502, html

    def http_error(url, body, timeout):
        raise urllib.error.HTTPError(url, 503, "Service Unavailable", {},
                                     io.BytesIO(html.encode("utf-8")))

    def exc_with_newlines(url, body, timeout):
        raise RuntimeError(f"first line\r\nsecond line https://api.telegram.org/bot{token}/x\n")

    for i, (stub, want) in enumerate([(non_2xx, "HTTP 502: <html> | <head><title>502 Bad Gateway"),
                                      (http_error, "HTTP 503: <html> | <head><title>502 Bad Gateway"),
                                      (exc_with_newlines, "RuntimeError: first line | second line")]):
        base = _fresh(f"alerts-flat-fail-{i}")
        alerts._post = stub
        with Env(token, "1"):
            out = alerts.notify(base, "halt", "rr-x halted by r1_profit_factor\nline two",
                                fields={"rule": "r1_profit_factor"})
        assert not out["pushed"] and out["telegram"]["sent"] is False
        err = out["telegram"]["error"]
        assert "\n" not in err and "\r" not in err and want in err, err
        assert token not in err and "<token>" in err, err
        text = (base / alerts.ALERTS_FILE).read_text(encoding="utf-8")
        lines = text.splitlines()
        assert len(lines) == 2 and text.endswith("\n"), lines
        assert lines[0].startswith("ALERT 20") and "[halt] rr-x halted by r1_profit_factor | line two" in lines[0]
        assert lines[1].startswith("ALERT 20") and "[telegram] push of [halt] failed after 2 attempt(s)" in lines[1]
        assert want in lines[1] and token not in text and "<token>" in lines[1], lines[1]
    # the heartbeat stores the same folded text and prints it on one line
    t0 = NOW
    base2 = _hb_base("hb-flat-fail", t0)
    alerts._post = non_2xx
    with Env(token, "1"):
        rep = risk.heartbeat(base2, now=t0)
    assert not rep["push"]["delivered"]
    doc = json.loads((base2 / risk.HEARTBEAT_FILE).read_text(encoding="utf-8"))
    last_error = doc["items"]["stale"]["last_error"]
    assert "\n" not in last_error and "HTTP 502: <html> | <head>" in last_error and token not in last_error
    printed = risk.format_heartbeat(rep)
    assert token not in printed
    for line in printed.splitlines():
        if "502" in line:
            assert "<html> | <head>" in line and "</body></html>" in line, line


def test_resume_in_another_process_is_honoured_by_a_poll_holding_the_run():
    import subprocess

    base = _fresh("resume-subprocess")
    _quiet_alerts()
    with Env(None, None):
        a = make_run(base, "res-sub", _declining(25, 100_000.0, 85_000.0), equity=85_000.0)
        assert a._risk_check() is not None and a.is_halted
        assert a.state["risk"]["tripped"] == "r2_equity_band" and "risk_epoch" not in a.state
        n_before = len(_records(a.root))
        # a person resumes it from a shell while object A still holds the run
        cmd = [sys.executable, os.path.join(ROOT, "paper.py"), "--dir", str(base),
               "resume", "--id", "res-sub", "--reason", "reviewed, carry on"]
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        env.pop(alerts.TOKEN_VAR, None)
        env.pop(alerts.CHAT_VAR, None)
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        on_disk = json.loads(a.state_path.read_text(encoding="utf-8"))
        ep = on_disk["risk_epoch"]
        assert ep["bar"] == a.state["last_bar"] and on_disk["risk"] is None
        assert not a.is_halted and a.state.get("risk_epoch") is None      # A knows nothing yet
        # A polls: no new bar on the csv, so the rules re-read the mirrored equity
        res = a.poll()
        assert "halt" not in res and res.get("halted") is None and not a.is_halted, res
        assert a.state["risk_epoch"] == ep
        assert a.state["risk"]["epoch_bars"] == 0 and a.state["risk"]["tripped"] is None
        assert a.state["risk"]["epoch"] == ep
        recs = _records(a.root)
        assert [r["type"] for r in recs[n_before:]] == ["resumed"]   # no second halt, no risk line
        saved = json.loads(a.state_path.read_text(encoding="utf-8"))
        assert saved["risk_epoch"] == ep and saved["risk"]["tripped"] is None
        # and a save from an object whose epoch is None does not clobber the file's
        a.state["risk_epoch"] = None
        a._save_state()
        assert json.loads(a.state_path.read_text(encoding="utf-8"))["risk_epoch"] == ep
        assert a.state["risk_epoch"] == ep
        # a halt written by hand in another process is seen by the next check too
        risk.write_halted(a.root, risk.make_marker("manual", None, None, {"reason": "x"}, ep))
        assert a._risk_check() is None and a.poll().get("halted") is True


def test_second_resume_in_another_process_is_not_clobbered_by_a_poll_holding_the_run():
    """The normal case after a false trip: the run was resumed once already, so
    the object holding it has a non-None epoch (ep1). A second `resume` in
    another shell writes ep2 and removes the marker while a poll is in flight;
    the tail of that poll is a save, which must keep ep2 — a save of ep1 over
    it re-trips the next poll on the numbers the person just cleared."""
    import subprocess

    base = _fresh("resume-subprocess-2")
    _quiet_alerts()
    with Env(None, None):
        # steep enough to cross the band from an epoch that starts at forward bar 5,
        # whatever BAND_SIGMAS is up to 3 (3 x ~0.0095 x sqrt(25) is about -14 %)
        run = make_run(base, "res-sub2", _declining(30, 100_000.0, 78_000.0), equity=78_000.0)
        bars = [r for r in _records(run.root) if r["type"] == "bar"]
        # an earlier resume at forward bar 5: an epoch that is not None in memory
        ep1 = {"bar": bars[4]["bar"], "equity": float(bars[4]["equity"]), "closed_trades_before": 0}
        run.state["risk_epoch"] = ep1
        run._save_state()
        a = P.PaperRun.load(base, "res-sub2")
        assert a.state["risk_epoch"] == ep1
        halt = a._risk_check()
        assert halt and halt["rule"] == "r2_equity_band" and halt["epoch"] == ep1 and a.is_halted
        n_before = len(_records(a.root))
        cmd = [sys.executable, os.path.join(ROOT, "paper.py"), "--dir", str(base),
               "resume", "--id", "res-sub2", "--reason", "second look"]
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        env.pop(alerts.TOKEN_VAR, None)
        env.pop(alerts.CHAT_VAR, None)
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        on_disk = json.loads(a.state_path.read_text(encoding="utf-8"))
        ep2 = on_disk["risk_epoch"]
        assert ep2 != ep1 and ep2["bar"] == a.state["last_bar"] and on_disk["risk"] is None
        assert not a.is_halted and a.state["risk_epoch"] == ep1          # A still holds ep1
        # the in-flight poll ends with a save: ep2 stays on disk, A adopts it
        a._save_state()
        saved = json.loads(a.state_path.read_text(encoding="utf-8"))
        assert saved["risk_epoch"] == ep2 and saved["risk"] is None, saved["risk_epoch"]
        assert a.state["risk_epoch"] == ep2 and a.state["risk"] is None
        # and the next poll evaluates against ep2: nothing armed, no re-trip
        res = a.poll()
        assert "halt" not in res and res.get("halted") is None and not a.is_halted, res
        assert a.state["risk"]["epoch"] == ep2 and a.state["risk"]["epoch_bars"] == 0 \
            and a.state["risk"]["tripped"] is None
        assert [r["type"] for r in _records(a.root)[n_before:]] == ["resumed"]
        saved = json.loads(a.state_path.read_text(encoding="utf-8"))
        assert saved["risk_epoch"] == ep2 and saved["risk"]["tripped"] is None
        # the writer does not clobber itself: an in-process resume of a hand
        # halt puts its own, newer epoch on disk over ep2
        risk.write_halted(a.root, risk.make_marker("manual", None, None, {"reason": "y"}, ep2))
        assert a.poll().get("halted") is True
        a.state["equity"] = float(a.state["equity"]) - 1.0      # so the new epoch differs
        rec = a.resume("third look")
        ep3 = rec["epoch"]
        assert ep3 != ep2 and a.state["risk_epoch"] == ep3
        assert json.loads(a.state_path.read_text(encoding="utf-8"))["risk_epoch"] == ep3
        a._save_state()
        assert json.loads(a.state_path.read_text(encoding="utf-8"))["risk_epoch"] == ep3
        assert a.state["risk_epoch"] == ep3


def test_risk_evaluate_errors_fail_closed_after_three_in_a_row():
    base = _fresh("risk-error")
    _quiet_alerts()
    real = P.risk_mod.evaluate

    def boom(run_, now=None, seed=None):
        raise RuntimeError("boom: bars.csv unreadable")

    def boom2(run_, now=None, seed=None):
        raise ValueError("a different failure")

    with Env(None, None):
        run = make_run(base, "rerr", _declining(5, 100_000.0, 99_000.0), equity=99_000.0)
        n0 = len(_records(run.root))
        try:
            P.risk_mod.evaluate = boom
            assert run._risk_check() is None and not run.is_halted
            assert run.state["risk"]["error"].startswith("RuntimeError: boom") \
                and run.state["risk"]["consecutive_errors"] == 1
            assert run._risk_check() is None and not run.is_halted
            assert run.state["risk"]["consecutive_errors"] == 2
            errs = [r for r in _records(run.root) if r["type"] == "error"]
            assert len(errs) == 1 and errs[0]["stage"] == "risk" \
                and errs[0]["consecutive_errors"] == 1              # same text: one line
            # a successful evaluation resets the count and the next error is journaled again
            P.risk_mod.evaluate = real
            assert run._risk_check() is None and "rules" in run.state["risk"]
            P.risk_mod.evaluate = boom
            assert run._risk_check() is None and run.state["risk"]["consecutive_errors"] == 1
            assert len([r for r in _records(run.root) if r["type"] == "error"]) == 2
            # a different text is journaled even inside a streak
            P.risk_mod.evaluate = boom2
            assert run._risk_check() is None and run.state["risk"]["consecutive_errors"] == 2
            assert len([r for r in _records(run.root) if r["type"] == "error"]) == 3
            # the third in a row halts, with rule risk_error, the text and the count
            halt = run._risk_check()
            assert halt and halt["rule"] == "risk_error" and run.is_halted
            assert halt["inputs"]["consecutive_errors"] == 3 and halt["inputs"]["halt_after"] == 3
            assert halt["inputs"]["error"].startswith("ValueError: a different failure")
            marker = run._halted_marker()
            assert marker["rule"] == "risk_error" and marker["inputs"] == halt["inputs"]
            assert marker["value"] is None and marker["threshold"] is None
            recs = _records(run.root)
            halted = [r for r in recs if r["type"] == "halted"]
            assert len(halted) == 1 and halted[0]["inputs"] == marker["inputs"]
            assert len([r for r in recs if r["type"] == "error"]) == 3   # no line for the repeat
            log = (base / alerts.ALERTS_FILE).read_text(encoding="utf-8")
            assert log.count("ALERT ") == 1 and "[halt]" in log and "risk_error" in log
            fields = json.loads(log[log.index(' {"') + 1:])
            assert fields["rule"] == "risk_error" and fields["inputs"] == marker["inputs"]
            # halted: not evaluated again, whatever evaluate would do
            assert run._risk_check() is None
        finally:
            P.risk_mod.evaluate = real
        assert len(_records(run.root)) - n0 == 3 + 1                     # 3 error lines + halted
        # a dry run with a failing evaluate writes nothing, halts nothing
        run2 = make_run(base, "rerr-dry", _declining(5, 100_000.0, 99_000.0), equity=99_000.0)
        run2.state["risk"] = {"at": "x", "error": "RuntimeError: boom: bars.csv unreadable",
                              "consecutive_errors": 2}
        before = {p.name: p.read_bytes() for p in run2.root.iterdir()}
        try:
            P.risk_mod.evaluate = boom
            run2._dry_run = True
            halt = run2._risk_check()
        finally:
            run2._dry_run = False
            P.risk_mod.evaluate = real
        assert halt and halt.get("would_halt") and halt["rule"] == "risk_error"
        assert not run2.is_halted
        assert {p.name: p.read_bytes() for p in run2.root.iterdir()} == before


def test_records_skips_a_malformed_trailing_line_with_one_alert():
    import contextlib
    import io

    base = _fresh("journal-cut")
    run = make_run(base, "cut", _declining(5, 100_000.0, 99_000.0), equity=99_000.0)
    good = run._records()
    # a crash mid-write: the last line is cut short and has no newline
    with run.journal_path.open("a", encoding="utf-8") as fh:
        fh.write('{"type": "bar", "at": "2026-09-12T00:00:00+00:00", "bar": "2010-')
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        recs = run._records()
        recs2 = run._records()
        rt = run.round_trips()                       # the rules read through it too
    assert recs == good and recs2 == good and rt is not None
    text = err.getvalue()
    assert text.count("ALERT ") == 1 and "trailing journal line" in text \
        and "malformed" in text and "cut" in text, text
    # the next record does not get glued onto the cut line: the damage stays one line
    run._journal({"type": "note", "at": "x", "text": "after the crash"})
    raw = run.journal_path.read_text(encoding="utf-8").splitlines()
    assert raw[-2].startswith('{"type": "bar"') and raw[-1] == \
        json.dumps({"type": "note", "at": "x", "text": "after the crash"})
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        recs = run._records()
    assert recs == good + [{"type": "note", "at": "x", "text": "after the crash"}]
    assert err.getvalue() == ""                       # already reported by this object
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        fresh = P.PaperRun.load(base, "cut")
        assert fresh._records() == recs
    assert err.getvalue().count("ALERT ") == 1 and "journal line" in err.getvalue()
    # a healthy journal is untouched by the check: byte for byte
    run3 = make_run(base, "whole", _declining(3, 100_000.0, 99_500.0), equity=99_500.0)
    before = run3.journal_path.read_bytes()
    run3._close_open_journal_line()
    assert run3.journal_path.read_bytes() == before


def test_drop_pending_for_halt_on_hand_built_states():
    base = _fresh("drop-pending")
    run = make_run(base, "drop", _declining(5, 100_000.0, 99_000.0), equity=99_000.0)
    bar = run.state["last_bar"]
    n0 = len(_records(run.root))

    def unfilled_lines():
        return [r for r in _records(run.root) if r["type"] == "unfilled"]

    # routed None: nothing was ever sent (below min notional) — cleared, no record, no count
    run.state["pending"] = {"bar": bar, "target": 0.5, "signal": 1.0}
    run.state["inflight"] = {"bar": bar, "target": 0.5, "routed": None, "legs": [],
                             "why_not": "move is below the broker's minimum notional",
                             "order_style": "market"}
    run._drop_pending_for_halt()
    assert run.state["pending"] is None and run.state["inflight"] is None
    assert unfilled_lines() == [] and run.state.get("unfilled", 0) == 0
    assert len(_records(run.root)) == n0
    # routed False: the send failed for its own reason — that reason is the record's
    run.state["pending"] = {"bar": bar, "target": 0.5, "signal": 1.0}
    run.state["inflight"] = {"bar": bar, "target": 0.5, "routed": False, "legs": [],
                             "why_not": "market is closed for QQQ", "order_style": "market",
                             "planned_legs": []}
    run._drop_pending_for_halt()
    assert run.state["pending"] is None and run.state["inflight"] is None
    lines = unfilled_lines()
    assert len(lines) == 1 and lines[0]["unfilled"]["why_not"] == "market is closed for QQQ"
    assert lines[0]["unfilled"]["routed"] is False and lines[0]["bar"] == bar
    assert run.state["unfilled"] == 1
    # routed False without a why_not: the broker error, then a plain sentence
    run.state["pending"] = {"bar": bar, "target": 0.5, "signal": 1.0}
    run.state["inflight"] = {"bar": bar, "routed": False, "legs": [], "error": "BrokerError: 503"}
    run._drop_pending_for_halt()
    assert unfilled_lines()[-1]["unfilled"]["why_not"] == "BrokerError: 503"
    # a legacy pending with no inflight: the halt is the reason
    run.state["pending"] = {"bar": bar, "target": 0.5, "signal": 1.0}
    run.state["inflight"] = None
    run._drop_pending_for_halt()
    assert unfilled_lines()[-1]["unfilled"]["why_not"].startswith("halted — the run was halted")
    assert run.state["unfilled"] == 3
    # a market-on-open still waiting for its window
    run.state["pending"] = {"bar": bar, "target": 0.5, "signal": 1.0}
    run.state["inflight"] = {"bar": bar, "routed": True, "deferred": True, "legs": [],
                             "send_not_before": "a", "send_not_after": "b",
                             "order_style": "market_on_open", "planned_legs": []}
    run._drop_pending_for_halt()
    assert "market-on-open planned for the window a to b" in unfilled_lines()[-1]["unfilled"]["why_not"]
    assert run.state["inflight"] is None and run.state["unfilled"] == 4
    # legs at the broker: real, left in flight, nothing written
    run.state["pending"] = {"bar": bar, "target": 0.5, "signal": 1.0}
    run.state["inflight"] = {"bar": bar, "routed": True, "legs": [{"order_id": "1"}]}
    run._drop_pending_for_halt()
    assert run.state["pending"] is not None and run.state["inflight"] is not None
    assert run.state["unfilled"] == 4 and len(unfilled_lines()) == 4
    # an inflight for another bar says nothing about this pending
    run.state["inflight"] = {"bar": "other", "routed": None, "legs": []}
    run._drop_pending_for_halt()
    assert run.state["pending"] is None and run.state["inflight"] is not None
    assert unfilled_lines()[-1]["unfilled"]["why_not"].startswith("halted") \
        and run.state["unfilled"] == 5



def test_heartbeat_clocks_tolerate_the_timers_jitter():
    """The timer fires with up to 5 s of random delay and the stamps are whole
    seconds, so the pass 'fifteen minutes later' (or six hours later) can land a
    few seconds early. It must still count — not wait one more pass."""
    t0 = NOW + timedelta(seconds=4.4)
    base = _hb_base("hb-jitter", t0)
    calls = []
    outcome = {"pushed": False, "telegram": {"error": "TimeoutError: no response"}}

    def notify(base_dir, kind, message, fields=None, push=None):
        calls.append(kind)
        return dict(outcome)

    risk.heartbeat(base, now=t0, notify=notify)
    assert calls == ["stale"]
    # +5 and +10 minutes, each a little early or late: still inside the backoff
    for minutes, jitter in ((5, -2.1), (10, 1.3)):
        t = NOW + timedelta(minutes=minutes, seconds=4.4 + jitter)
        _book(base, t)
        _touch(base, "fresh", t - timedelta(seconds=60))
        rep = risk.heartbeat(base, now=t, notify=notify)
        assert len(calls) == 1 and rep["push"] is None, minutes
    # +15 minutes but 3.4 s EARLIER within its minute than the failing pass: retried
    t1 = NOW + timedelta(minutes=15, seconds=1.0)
    _book(base, t1)
    _touch(base, "fresh", t1 - timedelta(seconds=60))
    outcome.update(pushed=True)
    outcome.pop("telegram")
    rep = risk.heartbeat(base, now=t1, notify=notify)
    assert len(calls) == 2 and rep["push"]["delivered"] is True
    # six hours after the delivered push, again a few seconds early: repeated;
    # five minutes before that: suppressed
    t_early = t1 + timedelta(hours=6, minutes=-5, seconds=2.0)
    _book(base, t_early)
    _touch(base, "fresh", t_early - timedelta(seconds=60))
    risk.heartbeat(base, now=t_early, notify=notify)
    assert len(calls) == 2
    t2 = t1 + timedelta(hours=6, seconds=-3.0)
    _book(base, t2)
    _touch(base, "fresh", t2 - timedelta(seconds=60))
    risk.heartbeat(base, now=t2, notify=notify)
    assert len(calls) == 3

# ────────────────────────────── runner ──────────────────────────────

# ────────────────────────────── the sigma floor (§1.2 amendment, 2026-09-18) ──────────────────────────────

def _often_flat_bars(n_up: int = 110, n_down: int = 215, seed: int = 7) -> pd.DataFrame:
    """A rise, then a long decline: a long-only trend filter is invested for a
    stretch of the rise and flat for nearly all of the rest."""
    rng = np.random.default_rng(seed)
    rets = np.concatenate([rng.normal(0.004, 0.006, n_up), rng.normal(-0.003, 0.006, n_down)])
    close = 100 * np.exp(np.cumsum(rets))
    idx = pd.bdate_range("2020-01-01", periods=len(close), name="date")
    return pd.DataFrame({"open": close, "high": close * 1.002, "low": close * 0.998,
                         "close": close, "volume": 1e5}, index=idx)


def _floor_of(run: P.PaperRun) -> float:
    return run.config["vol_target"] / np.sqrt(data.periods_per_year(run.bars().index))


def test_r2_floor_applies_when_an_often_flat_seed_sits_below_it():
    base = _fresh("r2-floor")
    run = make_run(base, "flat", _declining(25, 100_000.0, 99_000.0), equity=99_000.0,
                   strategy="trend_filter", bars=_often_flat_bars())
    prof = risk.seed_profile(run)
    seed, inv = prof["returns"], prof["invested_returns"]
    assert len(inv) < len(seed) / 2, (len(inv), len(seed))       # flat most of the time
    sigma_seed, floor = float(seed.std(ddof=1)), _floor_of(run)
    assert 0 < sigma_seed < floor, (sigma_seed, floor)
    r2 = risk.evaluate(run)["rules"]["r2_equity_band"]
    i = r2["inputs"]
    assert abs(i["sigma_seed"] - sigma_seed) < 1e-7 and abs(i["sigma_floor"] - floor) < 1e-7, i
    assert i["sigma_source"] == "floor" and i["sigma_bar"] == i["sigma_floor"], i
    assert abs(i["sigma_floor"] - 0.15 / np.sqrt(252.0)) < 1e-7, i     # a daily business calendar
    assert r2["armed"] and abs(r2["threshold"] - (-risk.BAND_SIGMAS * floor * np.sqrt(25))) < 1e-6, r2
    # the floor only ever widens the band
    assert r2["threshold"] < -risk.BAND_SIGMAS * sigma_seed * np.sqrt(25)
    # a loss between the two bands: a trip on the seed sigma, healthy on the floored one
    n = 25
    between = 100_000.0 * float(np.exp(-risk.BAND_SIGMAS * np.sqrt(n) * (sigma_seed + floor) / 2))
    run = make_run(base, "between", _declining(n, 100_000.0, between), equity=between,
                   strategy="trend_filter", bars=_often_flat_bars())
    r2 = risk.evaluate(run)["rules"]["r2_equity_band"]
    assert r2["armed"] and not r2["trip"], r2
    assert r2["value"] < -risk.BAND_SIGMAS * sigma_seed * np.sqrt(n), r2       # the old band would have tripped
    # and below the floored band it still trips
    run = make_run(base, "below", _declining(n, 100_000.0, 85_000.0), equity=85_000.0,
                   strategy="trend_filter", bars=_often_flat_bars())
    assert risk.evaluate(run)["rules"]["r2_equity_band"]["trip"]


def test_r2_floor_does_not_apply_when_the_seed_sigma_is_above_it():
    base = _fresh("r2-nofloor")
    run = make_run(base, "bh", _declining(25, 100_000.0, 99_000.0), equity=99_000.0)
    seed = risk.seed_returns(run)
    sigma_seed, floor = float(seed.std(ddof=1)), _floor_of(run)
    assert sigma_seed > floor, (sigma_seed, floor)
    r2 = risk.evaluate(run)["rules"]["r2_equity_band"]
    i = r2["inputs"]
    assert i["sigma_source"] == "seed" and i["sigma_bar"] == i["sigma_seed"], i
    assert abs(i["sigma_bar"] - sigma_seed) < 1e-7 and abs(i["sigma_floor"] - floor) < 1e-7, i
    assert abs(r2["threshold"] - (-risk.BAND_SIGMAS * sigma_seed * np.sqrt(25))) < 1e-6, r2


def test_r2_no_floor_without_a_vol_target_and_the_floor_never_arms_the_rule():
    base = _fresh("r2-nofloor-vt")
    run = make_run(base, "novt", _declining(25, 100_000.0, 99_000.0), equity=99_000.0,
                   strategy="trend_filter", bars=_often_flat_bars(), vol_target=None)
    r2 = risk.evaluate(run)["rules"]["r2_equity_band"]
    i = r2["inputs"]
    assert i["sigma_floor"] is None and i["sigma_source"] == "seed", i
    assert i["sigma_bar"] == i["sigma_seed"] and i["sigma_bar"] > 0 and r2["armed"], r2
    assert risk.sigma_floor(None, run.bars()) is None
    assert risk.sigma_floor(0.15, None) is None
    # every R2 evaluation carries the three fields, armed or not
    for name in ("sigma_seed", "sigma_floor", "sigma_source"):
        assert name in i, name
    # a disabled rule stays disabled: a floor exists, there is no seed sigma,
    # and the floor does not stand in for it
    run = make_run(base, "short", _declining(25, 100_000.0, 99_000.0), equity=99_000.0,
                   n_seed=300, strategy="tsmom")
    r2 = risk.evaluate(run)["rules"]["r2_equity_band"]
    i = r2["inputs"]
    assert not r2["armed"] and not r2["trip"] and r2["threshold"] is None, r2
    assert i["sigma_bar"] is None and i["sigma_seed"] is None and i["sigma_source"] is None, i
    assert i["sigma_floor"] is not None and "rule disabled" in r2["note"], r2
    # a bare series through evaluate(seed=...) is floored the same way
    run = make_run(base, "bare", _declining(25, 100_000.0, 99_000.0), equity=99_000.0)
    tiny = pd.Series(np.random.default_rng(2).normal(0.0, 0.001, 200))
    i = risk.evaluate(run, seed=tiny)["rules"]["r2_equity_band"]["inputs"]
    assert i["sigma_source"] == "floor" and i["sigma_bar"] == i["sigma_floor"] > i["sigma_seed"], i


def test_false_trip_rate_takes_the_band_sigma_and_other_multiples():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0, 0.01, 400))
    own = risk.false_trip_rate(r, paths=300)
    # the default sigma is the sd of the returns passed in
    assert risk.false_trip_rate(r, paths=300, sigma=float(r.std(ddof=1))) == own
    assert risk.false_trip_rate(r, paths=300, sigmas=risk.BAND_SIGMAS) == own
    # a wider sigma (the floor) is crossed less often; so is a wider multiple
    floored = risk.false_trip_rate(r, paths=300, sigma=0.016)
    assert floored < own, (floored, own)
    two = risk.false_trip_rate(r, paths=300, sigma=0.012, sigmas=2.0)
    three = risk.false_trip_rate(r, paths=300, sigma=0.012, sigmas=3.0)
    assert three < two, (three, two)
    assert three == risk.false_trip_rate(r, paths=300, sigma=0.012, sigmas=3.0)   # deterministic
    assert risk.BAND_SIGMAS == 3.0                       # a report multiple, not the rule's
    # an explicit sigma makes a flat series computable (it never crosses); a bad one does not
    assert risk.false_trip_rate(pd.Series([0.0] * 50), sigma=0.01, paths=50) == 0.0
    assert risk.false_trip_rate(r, sigma=0.0) is None
    assert risk.false_trip_rate(r, sigma=float("nan")) is None


def test_invested_only_returns_exclude_the_flat_bars():
    base = _fresh("r2-invested")
    run = make_run(base, "flat", _declining(25, 100_000.0, 99_000.0), equity=99_000.0,
                   strategy="trend_filter", bars=_often_flat_bars())
    prof = risk.seed_profile(run)
    seed, inv = prof["returns"], prof["invested_returns"]
    res = _engine_returns(run)
    pos = res.position.fillna(0.0)
    expected = seed[pos.loc[seed.index] != 0]
    assert inv.equals(expected) and 0 < len(inv) < len(seed), (len(inv), len(seed))
    assert (pos.loc[inv.index] != 0).all()
    assert (pos.loc[seed.index.difference(inv.index)] == 0).all()
    # sitting out dilutes the marginal sd: the invested-only sd is the larger one
    assert float(inv.std(ddof=1)) > float(seed.std(ddof=1)) * 1.3
    # always invested: the two series are the same
    run = make_run(base, "bh", _declining(25, 100_000.0, 99_000.0), equity=99_000.0)
    prof = risk.seed_profile(run)
    assert prof["invested_returns"].equals(prof["returns"])
    # a disabled profile carries an empty one
    run = make_run(base, "short", _declining(25, 100_000.0, 99_000.0), equity=99_000.0,
                   n_seed=300, strategy="tsmom")
    prof = risk.seed_profile(run)
    assert prof["returns"].empty and prof["invested_returns"].empty


def test_cmd_risk_prints_the_floor_and_both_false_trip_rates_and_writes_nothing():
    import argparse
    import contextlib
    import hashlib
    import importlib.util
    spec = importlib.util.spec_from_file_location("paper_cli", os.path.join(ROOT, "paper.py"))
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    base = _fresh("r2-cli")
    make_run(base, "flat", _declining(25, 100_000.0, 99_000.0), equity=99_000.0,
             strategy="trend_filter", bars=_often_flat_bars())
    make_run(base, "novt", _declining(25, 100_000.0, 99_000.0), equity=99_000.0, vol_target=None)

    def digest() -> dict:
        return {str(p.relative_to(base)): hashlib.md5(p.read_bytes()).hexdigest()
                for p in sorted(base.rglob("*")) if p.is_file()}

    before = digest()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli.cmd_risk(argparse.Namespace(dir=str(base), id=None, all=True))
    text = buf.getvalue()
    assert rc == 0 and digest() == before                      # read-only
    assert "floor 0.009449 (vol_target / sqrt(ppy))" in text and "(floor)" in text, text
    assert "no floor (vol_target is None)" in text, text
    assert "seed returns," in text and "invested-only" in text, text
    assert "sigma_source=floor" in text and "sigma_source=seed" in text, text


def test_alert_malformed_token_is_not_configured_never_echoed_and_nothing_is_sent():
    """T-4 token hygiene. A token with a trailing CR / LF (a CRLF env file) or a
    space is not a token: urllib would refuse the URL it makes with a message
    that spells it out. It is NOT CONFIGURED — one alerts.log line says malformed,
    never the value — nothing is sent, and no spelling of it reaches stderr,
    alerts.log, the returned dict or an exception text."""
    import contextlib
    import urllib.parse
    good = "424242:TEST-ONLY-not-a-token-0123456789"
    assert alerts.token_ok(good) and alerts.TOKEN_SHAPE.pattern == r"^[0-9]{5,}:[A-Za-z0-9_-]{20,}$"
    for shape in ("1234:" + "a" * 30, "123456:" + "a" * 19, "123456:" + "a" * 20 + "!", "abcdef:" + "a" * 30,
                  "123456" + "a" * 30, good + "\n", good + "\r", " " + good, ""):
        assert not alerts.token_ok(shape), repr(shape)
    bads = [good + "\r", good + "\n", good + "\r\n", good + " ", " " + good, good[:20] + " " + good[20:],
            "   ", "\r\n"]
    for i, bad in enumerate(bads):
        base = _fresh(f"alerts-malformed-{i}")
        rec = Recorder(fail=AssertionError("transport must not be called"))
        alerts._post = rec
        alerts._malformed_told.clear()
        err = io.StringIO()
        with Env(bad, " 987654321\r\n"), contextlib.redirect_stderr(err):
            assert not alerts.configured() and alerts.credentials() == (None, None)
            outs = [alerts.notify(base, "halt", "rr-x halted"), alerts.notify(base, "fill", "second")]
            direct = alerts._send("never built")
            problem = alerts.token_problem()
        assert rec.calls == [] and direct["sent"] is False and direct["attempts"] == 0
        assert all(o["logged"] and not o["pushed"] for o in outs)
        log = (base / alerts.ALERTS_FILE).read_text(encoding="utf-8")
        blank = not bad.strip()
        assert ("not set" in problem) if blank else ("malformed" in problem)
        # ONE line says malformed (not one per alert), and the alerts themselves are still logged
        assert log.count("is malformed") == (0 if blank else 1) and log.count("ALERT ") == (2 if blank else 3), log
        assert "[halt] rr-x halted" in log and "[fill] second" in log
        for place in (err.getvalue(), log, json.dumps(outs), json.dumps(direct), problem):
            for form in (good, bad, bad.strip(), urllib.parse.quote(bad, safe=""), repr(bad)[1:-1]):
                assert len(form) < 6 or form not in place, (i, form)
    # the masks: raw, stripped and the percent-encoded form of each, in an exception text
    bad = good + "\r\n"
    with Env(bad, "1"):
        for leak in (f"InvalidURL: URL can't contain control characters. '/bot{bad!r}/sendMessage'",
                     f"boom https://api.telegram.org/bot{urllib.parse.quote(bad, safe='')}/sendMessage",
                     f"boom https://api.telegram.org/bot{good}/sendMessage", f"x {bad} y"):
            masked = alerts._mask(leak)
            assert good not in masked and urllib.parse.quote(bad, safe="") not in masked and "<token>" in masked, masked
    # a message that carries the token (a caller's bug) never reaches the log or stderr
    base = _fresh("alerts-malformed-msg")
    err = io.StringIO()
    alerts._post = Recorder()
    with Env(good, " 42 "), contextlib.redirect_stderr(err):
        assert alerts.configured() and alerts.credentials() == (good, "42")          # the chat id is stripped
        out = alerts.notify(base, "book", f"oops {good}")
        sent = alerts.notify(base, "halt", "ok")
    assert sent["pushed"] and alerts._post.calls[0]["body"]["chat_id"] == "42"
    assert good not in err.getvalue() and good not in (base / alerts.ALERTS_FILE).read_text(encoding="utf-8")
    assert good not in json.dumps(out) and good not in json.dumps(sent)
    alerts._malformed_told.clear()


def _main() -> int:
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    saved_post = alerts._post
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
        finally:
            alerts._post = saved_post
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
