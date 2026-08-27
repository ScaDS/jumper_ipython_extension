"""Sample a process tree while something runs, and export what it cost.

Every replay that cannot boot perfmonitor inside the runtime it measures ends up
in the same shape: a supervising Python process starts the monitor, the work
happens in a child, the child reports when the cell under test began and ended,
and those marks are turned into the one-cell history the shared reader already
understands. R replays work this way because perfmonitor cannot live inside R;
fork replays work this way because a monitor's sampler thread must not exist in
a process that is about to fork.

The timing is the only subtle part. Samples are stamped with ``perf_counter``
(CLOCK_MONOTONIC) while a child can only report wall-clock epoch seconds, so the
offset between the two clocks is measured once, here, in the process that owns
the sampler - and the child's marks are mapped onto the sampler's timeline.
"""
import json
import os
import time

import pandas as pd

from jumper_extension.adapters.ai_reviewer.benchmark.models import FAILED, OK
from jumper_extension.core.service import build_perfmonitor_magic_adapter

# The cell under test sits at index ``prefix_count`` in the replayed notebook;
# the reader slices the history positionally, so the target must land on that
# row. Placeholder rows stand in for the prefix cells the child already ran -
# their timings are never read, only the target row's window is.
TARGET_LANGUAGE_DEFAULT = "unknown"


def build_silent_adapter(where: str):
    """A magic adapter that monitors but never draws or prints.

    *where* names the process in the reasons a disabled feature reports, so a
    stray plot call explains itself instead of failing anonymously.
    """
    return build_perfmonitor_magic_adapter(
        plots_disabled=True,
        plots_disabled_reason=f"Plotting disabled in the benchmark {where}.",
        display_disabled=True,
        display_disabled_reason=f"Display disabled in the benchmark {where}.",
    )


def clock_offset() -> float:
    """How much to add to an epoch second to get the sampler's perf_counter.

    Measured back to back so the two reads refer to the same instant; over a
    benchmark's few seconds the two clocks do not drift apart meaningfully.
    """
    epoch = time.time()
    mono = time.perf_counter()
    return mono - epoch


def synthesize_history(
    prefix_count: int,
    target_code: str,
    language: str,
    start_epoch: float,
    end_epoch: float,
    offset: float,
) -> pd.DataFrame:
    """Build the cell history the child could not: one real row for the target.

    Prefix rows are placeholders whose windows are never sliced; only the target
    row carries the child's measured start/end, mapped onto the sampler clock so
    ``filter_perfdata`` selects exactly the samples taken while the cell ran.
    """
    rows = []
    for index in range(prefix_count):
        rows.append(
            {
                "cell_index": index,
                "cell_magics": [],
                "raw_cell": "",
                "language": language,
                "start_time": 0.0,
                "end_time": 0.0,
                "duration": 0.0,
                "wallclock_start_time": 0.0,
                "wallclock_end_time": 0.0,
            }
        )
    rows.append(
        {
            "cell_index": prefix_count,
            "cell_magics": [],
            "raw_cell": target_code,
            "language": language,
            "start_time": start_epoch + offset,
            "end_time": end_epoch + offset,
            "duration": end_epoch - start_epoch,
            "wallclock_start_time": start_epoch,
            "wallclock_end_time": end_epoch,
        }
    )
    return pd.DataFrame(rows)


def measure_session(
    adapter,
    interval: float,
    run,
    session_path: str,
    markers_path: str,
    prefix_count: int,
    target_code: str,
    language: str,
) -> dict:
    """Sample while *run* executes, then export a session built from its marks.

    *run* does whatever the language or the strategy needs and returns an outcome
    dict carrying at least ``status``; anything other than OK is passed straight
    back, because a run that did not finish has nothing worth exporting. On
    success the child must have left epoch marks at *markers_path*.

    The monitor is stopped before anything is read back, so no sampling overlaps
    the bookkeeping that follows it.
    """
    adapter.perfmonitor_start(str(interval))
    offset = clock_offset()
    try:
        outcome = run()
    finally:
        adapter.perfmonitor_stop("")

    if outcome.get("status") != OK:
        return outcome
    if not os.path.exists(markers_path):
        return {
            "status": FAILED,
            "error": "The run left no timing marks for the cell under test.",
        }

    with open(markers_path) as handle:
        markers = json.load(handle)
    adapter.service.cell_history.data = synthesize_history(
        prefix_count=prefix_count,
        target_code=target_code,
        language=language,
        start_epoch=float(markers["start"]),
        end_epoch=float(markers["end"]),
        offset=offset,
    )
    adapter.export_session(session_path)
    return outcome
