"""Remote monitoring agent that runs on each SLURM node.

This module is executed as ``python -m jumper_extension.monitor_slurm_multinode._agent``
on each remote node via SSH.  It instantiates the default
:class:`PerformanceMonitor`, collects metrics at the requested interval,
and writes one JSON object per sample to *stdout* (one line per object).

The orchestrator on the head node reads these lines to aggregate results.

Protocol (stdout, one JSON line per sample)::

    {"node": "<hostname>", "time": <float>, "level": "<level>", "sample": {…}}

A special ``{"status": "ready", "node": "<hostname>", ...}`` line is
emitted once the monitor is initialised so the orchestrator knows the
agent is alive.

The agent stops gracefully when *stdin* is closed or when it receives
a SIGTERM / SIGINT.
"""

import json
import os
import signal
import socket
import sys
import time
from typing import List, Optional

# Make sure the package is importable even when invoked stand-alone on
# the remote node.  The orchestrator ensures the correct PYTHONPATH.
from jumper_extension.monitor.common import PerformanceMonitor


def _run_agent(interval: float, levels: Optional[List[str]] = None) -> None:
    hostname = socket.gethostname()
    monitor = PerformanceMonitor()

    if levels is None:
        levels = monitor.levels

    # Emit "ready" handshake
    ready_msg = {
        "status": "ready",
        "node": hostname,
        "num_cpus": monitor.num_cpus,
        "num_system_cpus": monitor.num_system_cpus,
        "num_gpus": monitor.num_gpus,
        "gpu_memory": monitor.gpu_memory,
        "gpu_name": monitor.gpu_name,
        "levels": levels,
        "pid": os.getpid(),
    }
    sys.stdout.write(json.dumps(ready_msg) + "\n")
    sys.stdout.flush()

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

    try:
        while running:
            t0 = time.perf_counter()
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
                    "node": hostname,
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

            elapsed = time.perf_counter() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)
    except BrokenPipeError:
        pass
    finally:
        monitor.running = False


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="JUmPER remote node monitoring agent"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default=None,
        help="Comma-separated list of levels to monitor (default: all available)",
    )
    args = parser.parse_args()
    levels = args.levels.split(",") if args.levels else None
    _run_agent(args.interval, levels)


if __name__ == "__main__":
    main()
