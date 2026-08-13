#!/usr/bin/env python3
"""Time the same cells under every benchmark replay mode, and write down every run.

The question is whether `fork` and `dill` are comparable with `full` *as
instruments*, so this drives `BenchmarkRunner` directly: no model, no
orchestrator, no repair loop. What that leaves is exactly what is being compared -
one prefix, one cell, N measurements per mode.

Two design points do the actual work here:

**Modes are interleaved, not run in blocks.** A ratio between two blocks of
measurements also contains whatever the machine's load did between them. The
three runners are therefore alive at the same time and take turns, with the order
rotated each round - the arrangement worklog SS4.4 used, widened from a flip to a
rotation so no mode keeps a position. Each fast mode's setup (the fork zygote,
the dill checkpoint) survives between turns, so their whole point - the prefix
paid once - is preserved.

**Degradation is detected, not assumed.** `resolve_strategy` falls back to the
full replay with a warning whenever the requested mode cannot serve, and
`BenchmarkRunner.fall_back` does the same mid-run. Neither raises. Without
checking which strategy actually ran, this script could plot three columns of
which two are secretly full - so every row records the strategy that ran. One
case (`open_handle`) is written to trigger exactly that, because a refusal is the
correct behaviour there.

Usage:
    python run_experiment.py                       # all cases, 5 runs each
    python run_experiment.py --runs 2 --cases array_reduce
"""
import argparse
import json
import logging
import os
import platform
import re
import shutil
import signal
import statistics
import sys
import tempfile
import time
import zipfile
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from jumper_extension.adapters.ai_reviewer.benchmark.replay import (
    DILL,
    FORK,
    FULL,
    StrategyChanged,
)
from jumper_extension.adapters.ai_reviewer.benchmark.runner import BenchmarkRunner
from jumper_extension.core.service import build_perfmonitor_service

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cases import CASES, Case  # noqa: E402  (path has to be set up first)

HERE = Path(__file__).resolve().parent
# full first, so it reads as the reference the other two are measured against.
MODES = (FULL, FORK, DILL)
FAST_MODES = (FORK, DILL)

# The extension's own default for benchmark replays (ai.benchmark.interval), so
# the sampler here behaves as it does in a real review.
DEFAULT_INTERVAL = 0.05
DEFAULT_RUNS = 5
DEFAULT_TIMEOUT_S = 300.0
DEFAULT_PREPARE_TIMEOUT_S = 120.0
# Coarse next to the measurements themselves: this samples the harness, which
# spends whole seconds per measurement, and a fast sampler would compete with
# what is being timed.
DEFAULT_PROFILE_INTERVAL = 0.25
# Prefix for virtual cells recorded by ``PerfmonitorService.monitored``. The
# report groups these cells by case and requested replay mode.
PROFILE_CELL_PREFIX = "# benchmark profile: "
PROFILE_SESSION_NAME = "harness_session.zip"

# The extension instance profiling this script, or None when --profile was not
# asked for.
SERVICE = None

# The fault count only exists inside the warning text: `_warn_if_distorted`
# states it and returns nothing.
_FAULTS = re.compile(r"on (\d+) page faults")
_PROBE = re.compile(r"benchmark fork probe: (.*)")
# Same for what the dill mode found out about itself while preparing: the
# checkpoint's size is logged at DEBUG and never returned.
_CHECKPOINT = re.compile(r"benchmark dill checkpoint: (\d+) bytes")


def profile_block(case: str, mode: str, phase: str):
    """Return a named JUmPER context, or a no-op when profiling is disabled."""
    if SERVICE is None:
        return nullcontext()
    return SERVICE.monitored(
        raw_cell=f"{PROFILE_CELL_PREFIX}{case} | {mode} | {phase}",
        should_skip_report=True,
    )


class LogCapture(logging.Handler):
    """Read the two answers that are only ever logged.

    The per-measurement page-fault estimate and the fork probe's verdict never
    reach a return value: the strategy warns about the first and logs the second
    at DEBUG. Both are precisely what this experiment is about, so they are taken
    off the logger rather than by reaching into the strategy's internals.
    """

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.lines: list[str] = []
        self._logger = logging.getLogger("extension")
        self._previous_level = self._logger.level

    def __enter__(self):
        self._logger.addHandler(self)
        return self

    def __exit__(self, *exception):
        self._logger.removeHandler(self)
        self._logger.setLevel(self._previous_level)

    def emit(self, record: logging.LogRecord):
        message = record.getMessage()
        # Everything the extension addresses to the user is prefixed; the rest is
        # its internal debug chatter, which would drown the two lines wanted here.
        if "[JUmPER]" in message:
            self.lines.append(message)

    def take(self) -> list[str]:
        """Lines logged since the last call, and clear."""
        lines, self.lines = self.lines, []
        return lines

    def at_debug(self):
        """Raise the level for one block - the probe verdict is logged at DEBUG."""
        return _Level(self._logger, logging.DEBUG)


class Watchdog:
    """Bound a call that has no timeout of its own.

    Setting a strategy up can block with nothing to notice: the fork chain waits
    on the zygote's answer while polling that it is still alive, and a zygote
    whose child deadlocked is alive and silent for as long as anyone cares to
    wait. That is an outcome worth recording, not a reason for the experiment to
    stop, so it is given a deadline here.
    """

    def __init__(self, seconds: float):
        self.seconds = seconds

    def __enter__(self):
        self._previous = signal.signal(signal.SIGALRM, self._expire)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, *exception):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self._previous)

    def _expire(self, *_):
        raise TimeoutError(f"did not return within {self.seconds:.0f}s")


class _Level:
    def __init__(self, logger: logging.Logger, level: int):
        self._logger = logger
        self._level = level
        self._previous = logger.level

    def __enter__(self):
        self._logger.setLevel(self._level)

    def __exit__(self, *exception):
        self._logger.setLevel(self._previous)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"measurements per mode per target (default {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--cases",
        default="",
        help=f"comma-separated subset of {','.join(CASES)} (default: all)",
    )
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="per-measurement budget; a runaway cell is killed past it. Cases "
        "that declare their own timeout_s override it",
    )
    parser.add_argument(
        "--prepare-timeout",
        type=float,
        default=DEFAULT_PREPARE_TIMEOUT_S,
        help="how long a strategy may take to set itself up before the mode is "
        "recorded as unusable for that case",
    )
    parser.add_argument(
        "--out",
        default="",
        help="results directory (default: results/<timestamp> beside this script)",
    )
    parser.add_argument(
        "--fixture-dir",
        default=str(HERE / "results" / "fixtures"),
        help="where on-disk fixtures live; reused across experiments",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="monitor this script with the extension itself, one JUmPER cell per "
        f"case, and write {PROFILE_SESSION_NAME} into the results directory",
    )
    parser.add_argument(
        "--profile-interval",
        type=float,
        default=DEFAULT_PROFILE_INTERVAL,
        help=f"sampling interval for --profile (default {DEFAULT_PROFILE_INTERVAL})",
    )
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help="keep the session exports and replay scripts each measurement left",
    )
    return parser.parse_args(argv)


def prefix_seconds(session_path: Path, target_index: int) -> float | None:
    """What re-running the prefix cost *this* measurement, from its own session.

    Every replay exports the history the hooks recorded, so the prefix cells are
    timed there already and nothing new has to be instrumented. Under `full` those
    rows are real: the prefix ran again for this measurement. Under `fork` and
    `dill` they are the placeholders a synthesized history carries, and sum to
    zero - which is the truth being measured, not a missing number: the prefix ran
    once, before any measurement, and is charged to `prepare_s` instead. What
    `dill` still pays per measurement is a restore rather than a replay, and that
    is read separately by `restore_seconds`.
    """
    try:
        with zipfile.ZipFile(session_path) as archive:
            with archive.open("cell_history.csv") as handle:
                cells = pd.read_csv(handle)
    except (OSError, KeyError, zipfile.BadZipFile):
        return None
    if "cell_index" not in cells or "duration" not in cells:
        return None
    prefix = cells[cells["cell_index"] < target_index]["duration"]
    return round(float(prefix.sum()), 6)


def machine() -> dict:
    memory = psutil.virtual_memory()
    return {
        "platform": platform.platform(),
        "kernel": platform.release(),
        "python": sys.version.split()[0],
        "cores_logical": psutil.cpu_count(logical=True),
        "cores_physical": psutil.cpu_count(logical=False),
        "memory_total_gb": round(memory.total / 1024**3, 1),
        "memory_available_gb": round(memory.available / 1024**3, 1),
        "numpy": np.__version__,
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", ""),
    }


def ensure_fixture(case: Case, fixture_dir: Path) -> str:
    """Lay the case's on-disk data down once, and reuse it next time.

    Built here rather than in the prefix because the full replay re-runs the
    prefix per measurement: a prefix that wrote 200MB would have every `full`
    number contain a disk write that the `fork` numbers never pay.
    """
    if not case.fixture_elements:
        return ""
    fixture_dir.mkdir(parents=True, exist_ok=True)
    path = fixture_dir / f"{case.name}_{case.fixture_elements}.f64"
    expected = case.fixture_elements * 8
    if path.exists() and path.stat().st_size == expected:
        return str(path)
    print(f"  building fixture {path.name} ({expected / 1024**2:.0f}MB)")
    mapping = np.memmap(path, dtype="float64", mode="w+", shape=(case.fixture_elements,))
    chunk = 4_000_000
    for start in range(0, case.fixture_elements, chunk):
        end = min(start + chunk, case.fixture_elements)
        mapping[start:end] = np.arange(start, end, dtype="float64")
    mapping.flush()
    del mapping
    return str(path)


def prepare_runner(
    case: Case,
    prefix_cells: list[dict],
    mode: str,
    args: argparse.Namespace,
    work_root: Path,
    capture: LogCapture,
) -> tuple[BenchmarkRunner, dict]:
    """Build a runner and get its setup out of the way before anything is timed.

    Preparing eagerly is what separates the modes' costs honestly: `fork` pays
    for a supervisor, a zygote and the prefix here, `dill` for the prefix and a
    checkpoint written to disk, while `full` pays nothing here and the whole
    prefix inside every measurement. Left lazy, the zygote's boot and the
    checkpoint's write would land inside their first measurement's wall time and
    read as a slow cell.

    `_ensure_prepared` is private, and it is the only way to ask for that split;
    a public `run_once` would fold setup into a measurement.
    """
    work_dir = work_root / f"{case.name}__{mode}"
    work_dir.mkdir(parents=True, exist_ok=True)
    runner = BenchmarkRunner(
        prefix_cells=prefix_cells,
        interval=args.interval,
        work_dir=str(work_dir),
        replay_mode=mode,
    )
    started = time.perf_counter()
    usable, failure = True, ""
    try:
        with Watchdog(args.prepare_timeout), capture.at_debug():
            runner._ensure_prepared()
    except TimeoutError as expired:
        usable, failure = False, f"setup {expired}"
    except Exception as error:
        usable, failure = False, f"{type(error).__name__}: {error}"
    prepare_s = round(time.perf_counter() - started, 4)

    lines = capture.take()
    probe, checkpoint_bytes = "", None
    for line in lines:
        found = _PROBE.search(line)
        if found:
            probe = found.group(1)
        sized = _CHECKPOINT.search(line)
        if sized:
            checkpoint_bytes = int(sized.group(1))
    # A mode that refused says so here rather than by failing: `_ensure_prepared`
    # swaps in the full replay and carries on, which is the behaviour under test
    # in `open_handle` - so the reason is kept even though nothing went wrong.
    refusal = ""
    if usable and runner.strategy.name != mode:
        refusal = next(
            (line for line in lines if "unusable" in line or "unavailable" in line),
            f"{mode} was replaced by {runner.strategy.name} without saying why",
        )
    return runner, {
        "mode": mode,
        "strategy": runner.strategy.name if usable else "unusable",
        "usable": usable,
        "failure": failure,
        "refusal": refusal,
        "prepare_s": prepare_s,
        "probe": probe,
        "checkpoint_bytes": checkpoint_bytes,
        "prepare_log": lines,
    }


def restore_seconds(runner: BenchmarkRunner, tag: str) -> float | None:
    """What loading the checkpoint cost *this* measurement, under `dill`.

    The dill mode's answer to "what does rebuilding state cost per measurement" -
    the counterpart of `prefix_seconds` under `full`, and of the flat zero under
    `fork`. It already times itself, to decide whether to warn that the mode is
    buying nothing, and writes the figure beside the checkpoint; it is read from
    there rather than re-derived, and the private attribute holding the directory
    is the only way to find it.
    """
    state_dir = getattr(runner.strategy, "_state_dir", "")
    if not state_dir:
        return None
    try:
        report = json.loads(Path(state_dir, f"{tag}.restore.json").read_text())
    except (OSError, ValueError):
        return None
    seconds = report.get("restore_s")
    return float(seconds) if seconds is not None else None


def measure(
    runner: BenchmarkRunner,
    case: Case,
    label: str,
    code: str,
    mode: str,
    run_index: int,
    position: int,
    args: argparse.Namespace,
    capture: LogCapture,
) -> dict:
    """One measurement, recorded whether it worked or not.

    A failed measurement is data: `thread_pool` under `fork` may well take a
    runtime that is not fork-safe with it, and that outcome belongs in the report
    rather than in a traceback.
    """
    tag = f"{case.name}_{label}_{mode}_{run_index}"
    capture.take()
    started = time.perf_counter()
    restore_s = None
    try:
        outcome = runner.run_once(code, tag, timeout=case.timeout_s or args.timeout)
        status, error = outcome.status, outcome.error
        duration_s, wall_s = outcome.duration_s, outcome.wall_s
        metrics, prints = outcome.metrics, outcome.fingerprints
        prefix_s = prefix_seconds(
            Path(runner.work_dir) / f"{tag}_session.zip",
            runner.strategy.target_cell_index,
        )
        restore_s = restore_seconds(runner, tag)
    except StrategyChanged as change:
        # The strategy gave out mid-run and the runner already swapped in the
        # full replay. In a real benchmark this unwinds the whole run; here it is
        # the finding, so it is recorded and the loop carries on.
        status, error = "strategy_changed", str(change)
        duration_s, wall_s, prefix_s = None, round(time.perf_counter() - started, 4), None
        metrics, prints = {}, {}
    except Exception as failure:  # a harness bug must not lose the other cases
        status, error = "harness_error", f"{type(failure).__name__}: {failure}"
        duration_s, wall_s, prefix_s = None, round(time.perf_counter() - started, 4), None
        metrics, prints = {}, {}

    # Timed here as well as by the strategy, because `run_once` returns wall time
    # only for a measurement that worked - and the cost of one that did *not* is
    # exactly what a mode that hangs should be charged for.
    harness_wall_s = round(time.perf_counter() - started, 4)
    warnings = capture.take()
    # Only the warning carries the count, and it is only stated when the faults
    # plausibly account for 5% of the measurement. Absent therefore means "not
    # reported", which is not the same as "none" - hence None rather than 0.
    faults = None
    for line in warnings:
        found = _FAULTS.search(line)
        if found:
            faults = int(found.group(1))
    return {
        "case": case.name,
        "target": label,
        "mode": mode,
        # What actually ran. A silent fallback would otherwise be invisible.
        "strategy": runner.strategy.name,
        "degraded": runner.strategy.name != mode,
        "run_index": run_index,
        "position": position,
        "status": status,
        "duration_s": duration_s,
        "prefix_s": prefix_s,
        "restore_s": restore_s,
        "wall_s": wall_s,
        "harness_wall_s": harness_wall_s,
        "faults_when_warned": faults,
        "metrics": json.dumps(metrics),
        "fingerprints": json.dumps(prints),
        "warnings": " | ".join(warnings),
        "error": (error or "")[:2000],
    }


def run_case(
    case: Case,
    args: argparse.Namespace,
    work_root: Path,
    capture: LogCapture,
) -> tuple[list[dict], dict]:
    """Every measurement for one case, with the two modes taking turns."""
    fixture_path = ensure_fixture(case, Path(args.fixture_dir))
    prefix_cells = case.prefix_cells(fixture_path)
    targets = case.resolved_targets(fixture_path)

    runners: dict[str, BenchmarkRunner] = {}
    setup: dict[str, dict] = {}
    records: list[dict] = []
    started = time.perf_counter()
    try:
        for mode in MODES:
            with profile_block(case.name, mode, "prepare"):
                runners[mode], setup[mode] = prepare_runner(
                    case,
                    prefix_cells,
                    mode,
                    args,
                    work_root,
                    capture,
                )
            size = setup[mode]["checkpoint_bytes"]
            print(
                f"  prepared {mode:>4}: strategy={setup[mode]['strategy']} "
                f"in {setup[mode]['prepare_s']}s"
                + (f"  probe: {setup[mode]['probe']}" if setup[mode]["probe"] else "")
                + (f"  checkpoint {size / 1024**2:.0f}MB" if size else "")
                + (f"  REFUSED -> full" if setup[mode]["refusal"] else "")
                + (f"  UNUSABLE: {setup[mode]['failure']}" if not setup[mode]["usable"] else "")
            )

        usable = [mode for mode in MODES if setup[mode]["usable"]]
        for mode in MODES:
            if mode in usable:
                continue
            # One row per target rather than none, so a mode that never got as far
            # as a measurement is visible in the data instead of only in the log.
            records.extend(
                {
                    "case": case.name,
                    "target": label,
                    "mode": mode,
                    "strategy": "unusable",
                    "degraded": True,
                    "run_index": 0,
                    "position": 0,
                    "status": "prepare_failed",
                    "duration_s": None,
                    "prefix_s": None,
                    "restore_s": None,
                    "wall_s": setup[mode]["prepare_s"],
                    "harness_wall_s": setup[mode]["prepare_s"],
                    "faults_when_warned": None,
                    "metrics": "{}",
                    "fingerprints": "{}",
                    "warnings": "",
                    "error": setup[mode]["failure"],
                }
                for label in targets
            )

        for run_index in range(args.runs):
            # Rotate which mode goes first each round, so a drift in machine load
            # cannot settle into one mode's column. With two modes a flip was
            # enough; with three, rotation is what gives each mode every position
            # equally often.
            offset = run_index % len(MODES)
            order = MODES[offset:] + MODES[:offset]
            for label, code in targets.items():
                for position, mode in enumerate(mode for mode in order if mode in usable):
                    phase = f"target={label} run={run_index + 1}"
                    with profile_block(case.name, mode, phase):
                        record = measure(
                            runners[mode],
                            case,
                            label,
                            code,
                            mode,
                            run_index,
                            position,
                            args,
                            capture,
                        )
                    records.append(record)
                    _print_measurement(record, args.runs)
    finally:
        for runner in runners.values():
            runner.close()

    return records, {
        "name": case.name,
        "aims_at": case.aims_at,
        "probes": case.probes,
        "expectation": case.expectation,
        "prefix": [cell["raw_cell"] for cell in prefix_cells],
        "targets": targets,
        "fixture": fixture_path,
        "setup": setup,
        "wall_s": round(time.perf_counter() - started, 2),
    }


def _print_measurement(record: dict, runs: int):
    cell = record["duration_s"]
    shown = f"{cell * 1000:9.2f}ms" if cell else "        --"
    wall = record["harness_wall_s"] or 0.0
    flag = "  DEGRADED" if record["degraded"] else ""
    note = "  faults=%d" % record["faults_when_warned"] if record["faults_when_warned"] else ""
    print(
        f"  {record['case']}/{record['target']:<10} {record['mode']:>4} "
        f"run {record['run_index'] + 1}/{runs}  {record['status']:<16} "
        f"cell {shown}  wall {wall:6.2f}s{note}{flag}"
    )


def summarize(records: list[dict]) -> pd.DataFrame:
    """Median per case/target/mode, and each fast mode's ratio against `full`.

    Reported two ways on purpose. The plain median uses every measurement; the
    one excluding the first run is what a real benchmark reports, because
    `runner.median_of` drops the first as warm-up - and neither a forked child
    nor a restored process is warmed by the run before it, both being new.
    """
    frame = pd.DataFrame(records)
    ok = frame[(frame["status"] == "ok") & frame["duration_s"].notna()]
    if ok.empty:
        return pd.DataFrame()

    rows = []
    for (case, target, mode), group in ok.groupby(["case", "target", "mode"]):
        ordered = group.sort_values("run_index")["duration_s"].tolist()
        # Dropped first: a median over nothing but NaN is not None, it is a numpy
        # warning printed into the middle of the report.
        faults = group["faults_when_warned"].dropna()
        restores = group["restore_s"].dropna() if "restore_s" in group else pd.Series(dtype=float)
        rows.append(
            {
                "case": case,
                "target": target,
                "mode": mode,
                "n": len(ordered),
                "restore_median_ms": 1000 * restores.median() if not restores.empty else None,
                "mean_ms": 1000 * statistics.fmean(ordered),
                "median_ms": 1000 * statistics.median(ordered),
                "median_excl_warmup_ms": 1000
                * statistics.median(ordered[1:] if len(ordered) > 1 else ordered),
                "min_ms": 1000 * min(ordered),
                "max_ms": 1000 * max(ordered),
                "wall_total_s": group["harness_wall_s"].sum(),
                "faults_when_warned_median": faults.median() if not faults.empty else None,
            }
        )
    summary = pd.DataFrame(rows).round(4)

    ratios = []
    for (case, target), group in summary.groupby(["case", "target"]):
        medians = group.set_index("mode")["median_ms"]
        if FULL not in medians or not medians[FULL]:
            continue
        ratios.append(
            {
                "case": case,
                "target": target,
                **{
                    f"ratio_{mode}_over_full": round(medians[mode] / medians[FULL], 3)
                    for mode in FAST_MODES
                    if mode in medians
                },
            }
        )
    if ratios:
        summary = summary.merge(pd.DataFrame(ratios), on=["case", "target"], how="left")
    return summary


def main(argv: list[str] | None = None) -> int:
    global SERVICE
    args = parse_args(argv)
    names = [name.strip() for name in args.cases.split(",") if name.strip()] or list(CASES)
    unknown = [name for name in names if name not in CASES]
    if unknown:
        print(f"unknown case(s): {', '.join(unknown)}; known: {', '.join(CASES)}")
        return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else HERE / "results" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    work_root = Path(tempfile.mkdtemp(prefix="replay-modes-"))

    print(f"{'/'.join(MODES)}: {len(names)} case(s), {args.runs} runs per mode")
    print(f"results -> {out_dir}")

    session_path = out_dir / PROFILE_SESSION_NAME
    if args.profile:
        # The default backend samples from a subprocess, so nothing is added to
        # this process that a case's own fork could inherit.
        SERVICE = build_perfmonitor_service(
            plots_disabled=True,
            display_disabled=True,
        )
        SERVICE.start_monitoring(interval=args.profile_interval)
        print(f"profiling this run every {args.profile_interval}s -> {session_path}")

    records: list[dict] = []
    case_meta: list[dict] = []
    interrupted = False
    profiled = False
    try:
        with LogCapture() as capture:
            for name in names:
                print(f"\n[{name}]")
                try:
                    case_records, meta = run_case(CASES[name], args, work_root, capture)
                except KeyboardInterrupt:
                    interrupted = True
                    print("  interrupted; writing what has been measured so far")
                    break
                except Exception as failure:
                    print(f"  case failed: {type(failure).__name__}: {failure}")
                    case_meta.append({"name": name, "error": f"{type(failure).__name__}: {failure}"})
                    continue
                records.extend(case_records)
                case_meta.append(meta)
    finally:
        # A monitor left running would outlive this process's own exit path, so
        # it is stopped even when the loop did not finish. Exported first: the
        # exporter refuses a session that is neither running nor imported.
        if SERVICE is not None:
            SERVICE.export_session(str(session_path))
            SERVICE.stop_monitoring()
            SERVICE = None
            profiled = session_path.exists()

    if not args.keep_work:
        shutil.rmtree(work_root, ignore_errors=True)
    else:
        print(f"\nwork directories kept in {work_root}")

    if not records:
        print("nothing was measured")
        return 1

    frame = pd.DataFrame(records)
    frame.to_csv(out_dir / "runs.csv", index=False)
    summary = summarize(records)
    if not summary.empty:
        summary.to_csv(out_dir / "summary.csv", index=False)

    meta = {
        "started_at": stamp,
        "finished_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "interrupted": interrupted,
        "args": vars(args),
        "machine": machine(),
        "cases": case_meta,
        # Named rather than inferred, so the report can tell "not profiled" from
        # "profiled and the export failed".
        "profile_session": PROFILE_SESSION_NAME if profiled else None,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n{len(frame)} measurements -> {out_dir / 'runs.csv'}")
    if not summary.empty:
        columns = ["case", "target", "mode", "n", "median_ms"]
        columns += [f"ratio_{mode}_over_full" for mode in FAST_MODES]
        print(summary[[column for column in columns if column in summary]].to_string(index=False))
    degraded = frame[frame["degraded"]]
    if not degraded.empty:
        by_mode = degraded.groupby("mode").size().to_dict()
        print(
            f"\nNOTE: {len(degraded)} measurement(s) ran on a fallback strategy "
            f"({by_mode}); those rows compare full against full. Expected for "
            "open_handle, where refusing is the correct behaviour - anywhere else, "
            "read the refusal reason in meta.json before reading the timings."
        )
    print(f"\nreport: open report.ipynb (it reads {out_dir.name} by default)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
