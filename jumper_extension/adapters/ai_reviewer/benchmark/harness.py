"""Language-agnostic replay harness: profile a child process, export a session.

This is the shared spine of Design B. For Python the replay script boots
perfmonitor and exports the session itself; for every other language that
machinery cannot live inside the language runtime, so it lives here instead.
The harness runs as its own process (the command an adapter's ``render_replay``
returns) and does only language-neutral work:

1. start perfmonitor, which samples this process and every child it spawns;
2. launch the language child (``--run``), which replays prefix + target cell
   in its own runtime and, around the target cell, writes two epoch timestamps
   (``--markers``) and a fingerprint of the cell's outputs;
3. stop perfmonitor and turn the child's timestamps into a one-cell history the
   shared reader already understands, then export the session the usual way.

The child's fingerprint file is written straight to the path the runner reads,
so the harness never touches it. The timing is the only subtle part: samples
are stamped with ``perf_counter`` (CLOCK_MONOTONIC) while the child can only
report wall-clock epoch seconds, so the harness measures the offset between the
two clocks once and maps the child's marks onto the sampler's timeline.
"""
import argparse
import json
import subprocess
import sys
import time

import pandas as pd

from jumper_extension.core.service import build_perfmonitor_magic_adapter

# The cell under test sits at index ``prefix_count`` in the replayed notebook;
# the reader slices the history positionally, so the target must land at that
# row. Placeholder rows stand in for the prefix cells the child already ran -
# their timings are never read, only the target row's window is.
_TARGET_LANGUAGE_DEFAULT = "unknown"


def _clock_offset() -> float:
    """How much to add to an epoch second to get the sampler's perf_counter.

    Measured back to back so the two reads refer to the same instant; over a
    benchmark's few seconds the two clocks do not drift apart meaningfully.
    """
    epoch = time.time()
    mono = time.perf_counter()
    return mono - epoch


def _synthesize_history(
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


def run_harness(
    run_cmd: list[str],
    session_path: str,
    markers_path: str,
    interval: float,
    prefix_count: int,
    target_code: str,
    language: str,
    work_dir: str,
) -> int:
    """Profile *run_cmd*, then export a session at *session_path*.

    Returns the child's exit code when it fails (so the runner reports the run
    as failed without a misleading session), otherwise 0 after a clean export.
    """
    adapter = build_perfmonitor_magic_adapter(
        plots_disabled=True,
        plots_disabled_reason="Plotting disabled in the benchmark harness.",
        display_disabled=True,
        display_disabled_reason="Display disabled in the benchmark harness.",
    )
    adapter.perfmonitor_start(str(interval))
    offset = _clock_offset()
    try:
        completed = subprocess.run(
            run_cmd,
            capture_output=True,
            text=True,
            cwd=work_dir,
        )
    finally:
        adapter.perfmonitor_stop("")

    if completed.stdout:
        sys.stdout.write(completed.stdout)
    if completed.returncode != 0:
        # Surface the child's own error so the runner's stderr tail is useful,
        # and skip the export: a failed replay has nothing worth comparing.
        sys.stderr.write(completed.stderr)
        return completed.returncode

    with open(markers_path) as handle:
        markers = json.load(handle)

    adapter.service.cell_history.data = _synthesize_history(
        prefix_count=prefix_count,
        target_code=target_code,
        language=language,
        start_epoch=float(markers["start"]),
        end_epoch=float(markers["end"]),
        offset=offset,
    )
    adapter.export_session(session_path)
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JUmPER benchmark replay harness")
    parser.add_argument("--run", required=True, help="child command as a JSON list")
    parser.add_argument("--session", required=True)
    parser.add_argument("--markers", required=True)
    parser.add_argument("--interval", type=float, required=True)
    parser.add_argument("--prefix-count", type=int, required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--target-code-file", default="")
    parser.add_argument("--language", default=_TARGET_LANGUAGE_DEFAULT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    target_code = ""
    if args.target_code_file:
        with open(args.target_code_file, encoding="utf-8") as handle:
            target_code = handle.read()
    return run_harness(
        run_cmd=json.loads(args.run),
        session_path=args.session,
        markers_path=args.markers,
        interval=args.interval,
        prefix_count=args.prefix_count,
        target_code=target_code,
        language=args.language,
        work_dir=args.work_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
