"""Replay an R cell under the monitor: run the language child, export a session.

The R half of Design B. Perfmonitor cannot live inside the R runtime, so it lives
here instead: this process starts the sampler, runs ``Rscript`` as a child that
replays prefix + target and writes epoch marks around the cell under test, and
hands those marks to the shared measurement spine to become a session export.

Everything language-neutral - the clocks, the synthesized history, the export -
lives in ``measure``; this module is only the R-specific entry point. The child's
fingerprint file is written straight to the path the runner reads, so nothing
here touches it.
"""
import argparse
import json
import subprocess
import sys

from jumper_extension.adapters.ai_reviewer.benchmark.measure import (
    TARGET_LANGUAGE_DEFAULT,
    build_silent_adapter,
    measure_session,
)
from jumper_extension.adapters.ai_reviewer.benchmark.models import FAILED, OK


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
    adapter = build_silent_adapter("harness")
    finished: dict = {}

    def run() -> dict:
        completed = subprocess.run(
            run_cmd,
            capture_output=True,
            text=True,
            cwd=work_dir,
        )
        finished["child"] = completed
        if completed.returncode != 0:
            return {"status": FAILED, "error": completed.stderr}
        return {"status": OK}

    outcome = measure_session(
        adapter=adapter,
        interval=interval,
        run=run,
        session_path=session_path,
        markers_path=markers_path,
        prefix_count=prefix_count,
        target_code=target_code,
        language=language,
    )

    child = finished.get("child")
    if child is not None and child.stdout:
        sys.stdout.write(child.stdout)
    if child is not None and child.returncode != 0:
        # Surface the child's own error so the runner's stderr tail is useful,
        # and skip the export: a failed replay has nothing worth comparing.
        sys.stderr.write(child.stderr)
        return child.returncode
    if outcome.get("status") != OK:
        sys.stderr.write(outcome.get("error", "the replay left no session behind"))
        return 1
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
    parser.add_argument("--language", default=TARGET_LANGUAGE_DEFAULT)
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
