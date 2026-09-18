"""tjr_human — the CLI of DESIGN-tjr-human.md section 9 (ticket T-7).

    python tjr_human.py detect                      the live loop on the TradingView feed; starts in OBSERVE mode
    python tjr_human.py arm --reason "..."          write ARMED: commit hash, SHA-256 of section 5, TARGET_RULE
    python tjr_human.py arm --disarm --reason "..." back to observe — refused once a trade has filled
    python tjr_human.py replay --csv NQ=f.csv --human none|script.json [--show-outcomes]
    python tjr_human.py status | report | review | final
    python tjr_human.py synth --out f.csv           a seeded random-walk file, flagged synthetic (for replay)

Paper only, and less than paper: a simulated ledger. No broker is constructed, no
order leaves the process, and the only network is Telegram — out through
quantlab.alerts.notify, in through getUpdates — and only when TELEGRAM_BOT_TOKEN
and TELEGRAM_CHAT_ID are in this process's environment (section 9.6: a launcher
the user keeps outside the repo). Nothing here reads, prints or asks for them.

`replay` measures the machine, not the strategy (section 9.8). On market data it
prints MACHINERY FACTS ONLY. `--show-outcomes` works on files flagged synthetic —
a `<file>.synthetic.json` sidecar bound to the file's bytes by SHA-256, written by
`synth` (or `runner.mark_synthetic`) — and refuses everything else.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from quantlab.tjr_human import N_TRIALS  # noqa: E402
from quantlab.tjr_human import journal as J  # noqa: E402
from quantlab.tjr_human import report as R  # noqa: E402
from quantlab.tjr_human import runner as RUN  # noqa: E402


def _pairs(values, what: str) -> dict[str, str]:
    """NQ=path (or a bare path whose file name starts with NQ / ES) -> {instrument: path}."""
    out = {}
    for v in values or []:
        if "=" in v:
            inst, path = v.split("=", 1)
        else:
            inst, path = Path(v).name[:2], v
        inst = inst.strip().upper()
        if inst not in RUN.INSTRUMENTS:
            raise SystemExit(f"{what}: cannot tell the instrument of {v!r}; write NQ=<path> or ES=<path>")
        if inst in out:
            raise SystemExit(f"{what}: {inst} given twice")
        if not Path(path).exists():
            raise SystemExit(f"{what}: {path} does not exist")
        out[inst] = path
    return out


def _records(base) -> list[dict]:
    return J.read_journal(base)


def _reportable(records) -> None:
    ok, why = RUN.journal_is_reportable(records)
    if not ok:
        raise SystemExit(f"REFUSED: {why}")


def cmd_detect(a) -> int:
    try:
        runner = RUN.live_runner(a.dir, method=a.feed_method, draw=not a.no_draw, accept_holes=a.accept_holes,
                                 backfill=_pairs(a.backfill, "--backfill"), keep_awake=not a.allow_sleep,
                                 accept_short_sessions=a.accept_short_session or ())
    except RUN.Refused as exc:
        print(f"REFUSED to start: {exc}", file=sys.stderr)
        return 2
    problem = RUN.telegram_problem()
    print(f"tjr_human detect: journal {a.dir}/, feed tradingview ({a.feed_method}), "
          f"mode {'ARMED' if runner.mgr.armed else 'OBSERVE (no entry is taken until `arm`)'}; "
          f"Telegram {'configured' if problem is None else 'OFF — ' + problem + ': alerts are logged only, no commands'}; "
          f"n_trials = {N_TRIALS}. Ctrl-C stops; the next start journals the gap.")
    try:
        runner.run_live(max_cycles=a.cycles)
    except RUN.Refused as exc:
        print(f"REFUSED to start: {exc}", file=sys.stderr)
        return 2
    return 0


def cmd_arm(a) -> int:
    try:
        rec = RUN.disarm(a.dir, a.reason) if a.disarm else RUN.arm(a.dir, a.reason, allow_dirty=a.allow_dirty)
    except RUN.Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    if a.disarm:
        print(f"DISARMED (seq {rec['seq']}, {rec['ts']}): observe mode from the next signal.")
        return 0
    print(f"ARMED (seq {rec['seq']}, {rec['ts']})\n  commit        {rec['commit']}"
          + ("   [uncommitted changes — journaled]" if rec.get("dirty") else "")
          + f"\n  section 5     sha256 {rec['prereg_sha256']}\n  TARGET_RULE   {rec['target_rule']}"
            f"\n  n_trials      {rec['n_trials']}; sample {rec['sample']}; pass bar {rec['pass_bar']}"
            f"\n  reason        {rec['reason']}\nFrom the next signal the bot takes entries and counts. "
            "Disarming is refused once a trade has filled.")
    return 0


def _print_facts(facts: dict) -> None:
    print("MACHINERY FACTS (section 9.8: counts of events only — no result of any kind; each line says its basis)")
    for inst, f in facts["instruments"].items():
        print(f"  {inst}: bars {f['bars']}, days routed {f['routed_days']}, cold days not routed "
              f"{f['cold_days_not_routed']}")
        print(f"  {inst} detector events [basis: {f['detector_events_basis']}]: {f['detector_events']}")
        routed = f["routed_sessions"]
        for key, label in (("detector_events", "detector events"),
                           ("sweeps_by_class_and_direction", "sweeps by level class / direction"),
                           ("confirmations_by_types", "confirmations by type combination"),
                           ("signals_by_zone", "signals by zone kind"),
                           ("entry_triggers_by_zone", "entry triggers by zone kind"),
                           ("invalidations_by_why_and_stage", "invalidations by reason @ stage"),
                           ("expiries_by_why", "expiries by reason")):
            print(f"  {inst} {label} [basis: {routed['basis']}]: {routed.get(key, {})}")
    for name in ("setups", "trades"):
        body = {k: v for k, v in facts[name].items() if k != "basis"}
        print(f"  {name} [basis: {facts[name]['basis']}]: {body}")
    c = facts["commands"]
    print(f"  commands: scripted {c['scripted']}, received {c['received']}, accepted {c['accepted']}, "
          f"rejected {c['rejected']} {c['rejected_by_rule']}; price observations {c['price_observations']}")
    j = facts["journal"]
    print(f"  journal: {j['records']} records {j['by_kind']}; integrity {'ok' if j['integrity_ok'] else 'BROKEN ' + str(j['problems'])}")
    print(f"  alerts: {facts['alerts']['notices']} notices {facts['alerts']['by_kind']} (replay: recorded, none sent)")
    d = facts["draws"]
    print(f"  draws: {d['asked']} shapes {d['by_shape']} (replay: chart layer disabled, none drawn)")


def cmd_replay(a) -> int:
    csvs = _pairs(a.csv, "--csv")
    if not csvs:
        raise SystemExit("replay needs --csv NQ=<file> and/or --csv ES=<file>")
    human = None if a.human in (None, "none") else a.human
    try:
        out = RUN.replay(csvs, human=human, base_dir=a.journal_dir, show_outcomes=a.show_outcomes,
                         armed=not a.observe, allow_cold=a.allow_cold, preview=not a.no_preview)
    except RUN.Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    for inst, s in out["sources"].items():
        print(f"{inst}: {Path(csvs[inst]).name} — {'SYNTHETIC' if s['synthetic'] else 'MARKET DATA'} ({s['why']})")
    if a.json:
        print(json.dumps(out["facts"], indent=2, default=str))
    else:
        _print_facts(out["facts"])
    if a.show_outcomes:
        print("\nOUTCOMES (synthetic data only)\n" + out["outcomes"])
    return 0 if out["facts"]["journal"]["integrity_ok"] else 1


def cmd_status(a) -> int:
    print(R.status(_records(a.dir), a.dir)["text"])
    return 0


def cmd_report(a) -> int:
    records = _records(a.dir)
    _reportable(records)
    print(R.weekly(records)["text"])
    return 0


def cmd_review(a) -> int:
    journal = J.Journal(a.dir)
    records = journal.records()
    _reportable(records)
    written = R.maybe_review(journal)
    reviews = J.rebuild(journal.records())["reviews"]
    if not reviews:
        filled = J.filled_count(records)
        print(f"tjr_human review: none yet — {filled} filled trades; reviews are written once each at "
              f"{' and '.join(str(n) for n in R.REVIEW_AT)}, when those trades are exited and benchmarked.")
        return 0
    for n, rec in sorted(reviews.items()):
        print(f"[review at {n}, journaled {rec['ts']}{' — written just now' if rec in written else ''}]")
        print(rec.get("text", ""))
        print()
    return 0


def cmd_final(a) -> int:
    journal = J.Journal(a.dir)
    _reportable(journal.records())
    out = R.final(journal)
    print(out["text"])
    return 0 if out.get("ran") else 2


def cmd_synth(a) -> int:
    frame = RUN.synthetic_walk(days=a.days, seed=a.seed, price=a.price)
    path = RUN.write_synthetic_csv(a.out, frame, generator=f"tjr_human.py synth --days {a.days} --seed {a.seed}")
    print(f"wrote {path} ({len(frame)} bars, {a.days} sessions) and {path.name}{RUN.MARKER_SUFFIX}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tjr_human.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=J.DEFAULT_DIR, help="journal directory (default tjr_human_runs)")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="the live loop; starts in OBSERVE mode")
    d.add_argument("--feed-method", choices=("eval", "focus"), default="eval",
                   help="eval: one `tv ui eval` reads every pane; focus: pane focus + state + ohlcv per instrument")
    d.add_argument("--no-draw", action="store_true", help="do not draw on the chart")
    d.add_argument("--accept-holes", action="store_true",
                   help="start although the two newest sessions of the store have a hole (journaled)")
    d.add_argument("--backfill", action="append", metavar="NQ=<csv>",
                   help="merge a 1-minute csv export into the store before starting (the store's own rows win)")
    d.add_argument("--accept-short-session", action="append", metavar="YYYY-MM-DD",
                   help="a trade a crash left open in this session ends at the session's last close: the session "
                        "closed early (a holiday) and the store will never hold its 15:55 bar (journaled)")
    d.add_argument("--allow-sleep", action="store_true",
                   help="do not ask Windows to stay awake while the loop runs (a sleeping PC is a hole in the store)")
    d.add_argument("--cycles", type=int, default=None, help="stop after this many polls (default: run until Ctrl-C)")
    d.set_defaults(fn=cmd_detect)

    m = sub.add_parser("arm", help="write ARMED (or, with --disarm, go back to observe)")
    m.add_argument("--reason", required=True)
    m.add_argument("--disarm", action="store_true")
    m.add_argument("--allow-dirty", action="store_true", help="arm with uncommitted changes (journaled)")
    m.set_defaults(fn=cmd_arm)

    r = sub.add_parser("replay", help="the same runner over stored 1-minute csv files (machinery facts only)")
    r.add_argument("--csv", action="append", metavar="NQ=<csv>")
    r.add_argument("--human", default="none", help="none, or a json list of timed commands")
    r.add_argument("--show-outcomes", action="store_true", help="synthetic files only; refuses market data")
    r.add_argument("--journal-dir", default=None, help="keep the replay journal here (synthetic files only)")
    r.add_argument("--observe", action="store_true", help="replay in observe mode (no entries)")
    r.add_argument("--allow-cold", action="store_true", help="route days whose levels lack the warm history")
    r.add_argument("--no-preview", action="store_true", help="closed bars only: the fill is known a minute late")
    r.add_argument("--json", action="store_true")
    r.set_defaults(fn=cmd_replay)

    for name, fn, text in (("status", cmd_status, "where the run is — nothing about whether it is winning"),
                           ("report", cmd_report, "the weekly report (section 3.6)"),
                           ("review", cmd_review, "interim reviews at 25 and 50: write the due one once, print them"),
                           ("final", cmd_final, f"the section 5 test at 100 filled trades (n_trials = {N_TRIALS})")):
        s = sub.add_parser(name, help=text)
        s.set_defaults(fn=fn)

    y = sub.add_parser("synth", help="write a synthetic 1-minute csv and its marker")
    y.add_argument("--out", required=True)
    y.add_argument("--days", type=int, default=25)
    y.add_argument("--seed", type=int, default=1)
    y.add_argument("--price", type=float, default=20000.0)
    y.set_defaults(fn=cmd_synth)

    a = p.parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    sys.exit(main())
