"""Local monitoring collector that runs in a child process.

This module is executed as
``python -m jumper_extension.monitor.backends.subprocess_python._collector``
by :class:`SubprocessPerformanceMonitor`.  It instantiates the default
:class:`PerformanceMonitor`, collects metrics at the requested interval,
and writes one JSON object per sample to *stdout* (one line per object).

The parent process reads these lines to populate its
:class:`PerformanceData` container.

Protocol (stdout, one JSON line per message)::

    {"status": "ready", "pid": <int>, ...}           # handshake
    {"level": "<level>", "time": <float>, "sample": {…}}  # data

The collector stops gracefully when *stdin* is closed (parent dies) or
when it receives SIGTERM / SIGINT.
"""

import json
import os
import signal
import sys
import time
from typing import List, Optional


def _run_collector(
    interval: float,
    levels: Optional[List[str]] = None,
    target_pid: Optional[int] = None,
) -> None:
    # Redirect noisy init output away from the JSON protocol channel
    import io
    import contextlib
    import logging

    log_capture = io.StringIO()
    log_handler = logging.StreamHandler(log_capture)
    log_handler.setLevel(logging.WARNING)

    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.addHandler(log_handler)
    root_logger.setLevel(logging.WARNING)

    temp_stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(temp_stdout):
            from jumper_extension.monitor.backends.thread import PerformanceMonitor
            monitor = PerformanceMonitor()
    except Exception as e:
        # Restore logging before writing error
        root_logger.removeHandler(log_handler)
        root_logger.setLevel(original_level)

        error_msg = {
            "status": "error",
            "pid": os.getpid(),
            "error": str(e),
        }
        sys.stderr.write(f"[SubprocessCollector] init error: {e}\n")
        sys.stderr.flush()
        sys.stdout.write(json.dumps(error_msg) + "\n")
        sys.stdout.flush()
        return

    root_logger.removeHandler(log_handler)
    root_logger.setLevel(original_level)

    # Point the monitor at the parent kernel process instead of ourselves
    if target_pid is not None:
        import psutil
        monitor.pid = target_pid
        monitor.process = psutil.Process(target_pid)

    # Forward captured output to stderr so it doesn't break the protocol
    for label, buf in [("init logs", log_capture), ("init stdout", temp_stdout)]:
        text = buf.getvalue()
        if text:
            sys.stderr.write(f"[SubprocessCollector {label}] {text}")
            sys.stderr.flush()

    if levels is None:
        levels = monitor.levels

    # --- handshake ---
    ready_msg = {
        "status": "ready",
        "pid": os.getpid(),
        "num_cpus": monitor.num_cpus,
        "num_system_cpus": monitor.num_system_cpus,
        "num_gpus": monitor.num_gpus,
        "gpu_memory": monitor.gpu_memory,
        "gpu_name": monitor.gpu_name,
        "memory_limits": monitor.memory_limits,
        "cpu_handles": monitor.cpu_handles,
        "levels": levels,
    }
    sys.stdout.write(json.dumps(ready_msg) + "\n")
    sys.stdout.flush()

    # --- main collection loop ---
    running = True

    def _shutdown(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    monitor.interval = interval
    monitor.start_time = time.perf_counter()
    monitor.wallclock_start_time = time.time()
    monitor.running = True

    next_tick = time.perf_counter()

    try:
        while running:
            monitor.process_pids = monitor._get_process_pids()
            metrics = monitor._collect_metrics()

            for level, data_tuple in zip(monitor.levels, metrics):
                if level not in levels:
                    continue
                (
                    time_mark,
                    cpu_util,
                    memory,
                    gpu_util,
                    gpu_band,
                    gpu_mem,
                    io_counters,
                ) = data_tuple

                sample = {
                    "time": time_mark,
                    "wallclock": time.time(),
                    "level": level,
                    "sample": {
                        "cpu_util": cpu_util,
                        "memory": memory,
                        "gpu_util": gpu_util,
                        "gpu_band": gpu_band,
                        "gpu_mem": gpu_mem,
                        "io_counters": io_counters,
                    },
                }
                sys.stdout.write(json.dumps(sample) + "\n")
            sys.stdout.flush()

            # absolute-time anchored sleep (no GIL issues — own process)
            next_tick += interval
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()

    except BrokenPipeError:
        sys.stderr.write("[SubprocessCollector] Broken pipe — parent exited\n")
        sys.stderr.flush()
    except Exception as e:
        sys.stderr.write(f"[SubprocessCollector] Error in main loop: {e}\n")
        sys.stderr.flush()
    finally:
        monitor.running = False


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="JUmPER local subprocess monitoring collector"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--target-pid",
        type=int,
        default=None,
        help="PID of the process to monitor (default: self)",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default=None,
        help="Comma-separated list of levels to monitor (default: all available)",
    )
    args = parser.parse_args()
    levels = args.levels.split(",") if args.levels else None
    _run_collector(args.interval, levels, target_pid=args.target_pid)


if __name__ == "__main__":
    main()
