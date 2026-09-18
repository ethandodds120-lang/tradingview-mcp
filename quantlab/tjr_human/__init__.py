"""tjr_human — the human exit layer experiment (DESIGN-tjr-human.md section 9, ticket T-7).

Paper only, and not even that: no broker is constructed anywhere in this
package and no order leaves the process. The strategy underneath failed the
gauntlet and is tagged folklore; this package measures a person's exits on its
entries against ten mechanical exits, under the pre-registration of section 5.

    detector    closed 1-minute bars in, events out; no I/O, no clock of its own
    exits       the ten mechanical exits and the random stop, causal replay
    trade       one setup / trade state machine and the rules of the four human controls
    commands    SKIP, STOP, MOVE STOP, EXIT NOW and the reason codes; the Telegram inbound transport
    journal     the append-only JSONL journal, its schema, its readers
    report      weekly report, interim reviews at 25 and 50, the final test at 100, status
    chart       display layer through the `tv` CLI; may fail without consequence; off in replay
    runner      the loop: feed (TradingView live, or stored csv replay) -> detector -> alert ->
                commands -> trade -> journal; the 1-minute store; ARMED

Nothing here is registered as a strategy and nothing here touches paper.py,
risk.py, the registry, the engine or the gauntlet.
"""

#: section 5, second amendment: ten mechanical exits + four human controls
N_TRIALS = 14

from . import detector, exits                       # noqa: F401,E402
from . import commands, journal, report, trade      # noqa: F401,E402
from . import chart, runner                          # noqa: F401,E402
from .detector import TARGET_RULE, TARGET_RULES, Detector, check_warm, detect          # noqa: F401,E402
from .exits import EXIT_NAMES, MECHANICAL_EXITS, RANDOM_EXIT, benchmark_record        # noqa: F401,E402
