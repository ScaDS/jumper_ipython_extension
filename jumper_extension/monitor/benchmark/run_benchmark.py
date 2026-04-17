#!/usr/bin/env python3
"""Monitor backend benchmark.

Runs each monitor implementation (thread, subprocess_python, native_c)
at several sampling frequencies while a CPU-heavy workload saturates all
available cores.  Each configuration is repeated multiple times;
outliers are removed and mean ± std are reported.

Results are saved as CSV files and visualised with three plot types:

    A. Run-chart  – binary hit/miss step plot + moving-average overlay
    B. Cumulative success curve – actual vs ideal sample count over time
    C. Histogram / KDE of inter-arrival times – jitter & tail behaviour

Usage
-----
    python -m jumper_extension.monitor.benchmark.run_benchmark [--duration 60]

The script writes its outputs into the ``benchmark/results/`` directory
next to this file.
"""

import argparse
import atexit
import multiprocessing
import os
import signal
import sys
import time

import numpy as np
import pandas as pd
import psutil


def _cleanup_children():
    """Kill any remaining child processes on exit."""
    for child in psutil.Process().children(recursive=True):
        try:
            child.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

atexit.register(_cleanup_children)

# ---------------------------------------------------------------------------
# Monitor factories
# ---------------------------------------------------------------------------

BACKENDS = {
    "thread": lambda: _make_thread_monitor(),
    "subprocess_python": lambda: _make_subprocess_monitor(),
    "native_c": lambda: _make_native_c_monitor(),
}

FREQUENCIES = [1, 2, 4, 8, 16]  # Hz


def _make_thread_monitor():
    from jumper_extension.monitor.backends.thread import PerformanceMonitor
    return PerformanceMonitor()


def _make_subprocess_monitor():
    from jumper_extension.monitor.backends.subprocess_python import (
        SubprocessPerformanceMonitor,
    )
    return SubprocessPerformanceMonitor()


def _make_native_c_monitor():
    from jumper_extension.monitor.backends.native_c import (
        CSubprocessPerformanceMonitor,
    )
    return CSubprocessPerformanceMonitor()


# ---------------------------------------------------------------------------
# CPU workload
# ---------------------------------------------------------------------------

def _cpu_burn(stop_event):
    """Pure-Python busy loop to saturate one core."""
    while not stop_event.is_set():
        s = 0
        for i in range(50_000):
            s += i * i


def _available_cpus():
    """Return the number of CPUs available to this process.

    Respects SLURM's ``SLURM_CPUS_PER_TASK`` / ``SLURM_CPUS_ON_NODE``,
    cgroup limits (``os.sched_getaffinity``), and falls back to
    ``os.cpu_count()`` only as a last resort.
    """
    # 1. SLURM environment (most reliable on shared HPC nodes)
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        val = os.environ.get(var)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                pass
    # 2. cgroup / taskset affinity
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    # 3. Fallback
    return os.cpu_count() or 4


def _print_cpu_diagnostics():
    """Print all CPU detection methods for debugging."""
    print("\nCPU detection diagnostics:")
    print(f"  SLURM_CPUS_PER_TASK:  {os.environ.get('SLURM_CPUS_PER_TASK', '<not set>')}")
    print(f"  SLURM_CPUS_ON_NODE:   {os.environ.get('SLURM_CPUS_ON_NODE', '<not set>')}")
    print(f"  SLURM_JOB_CPUS_PER_NODE: {os.environ.get('SLURM_JOB_CPUS_PER_NODE', '<not set>')}")
    try:
        aff = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        aff = '<unavailable>'
    print(f"  os.sched_getaffinity: {aff}")
    print(f"  os.cpu_count():       {os.cpu_count()}")
    print(f"  → _available_cpus():  {_available_cpus()}")
    print()


def start_workload(n_workers=None):
    """Spawn *n_workers* processes (default: available CPUs) doing busy work."""
    if n_workers is None:
        n_workers = _available_cpus()
    stop_event = multiprocessing.Event()
    workers = []
    for _ in range(n_workers):
        p = multiprocessing.Process(target=_cpu_burn, args=(stop_event,), daemon=True)
        p.start()
        workers.append(p)
    return workers, stop_event


def stop_workload(workers, stop_event):
    stop_event.set()
    # Send SIGTERM to all immediately — no need for graceful shutdown
    for p in workers:
        try:
            if p.is_alive():
                p.terminate()
        except (OSError, ValueError):
            pass
    # Give them a brief moment to exit, then force-kill stragglers
    for p in workers:
        try:
            p.join(timeout=0.5)
            if p.is_alive():
                p.kill()
        except (OSError, ValueError):
            pass
    for p in workers:
        try:
            p.join(timeout=0.5)
        except (OSError, ValueError):
            pass


# ---------------------------------------------------------------------------
# Experiment overview helpers
# ---------------------------------------------------------------------------

def _count_level_pids(monitor):
    """Return a dict mapping each active level to its PID count."""
    counts = {}
    for level in getattr(monitor, "levels", []):
        df = monitor.data.data.get(level, pd.DataFrame())
        counts[level] = len(df)
    return counts


def _snapshot_process_counts():
    """Snapshot process counts right now (call while workload is running)."""
    from jumper_extension.utilities import is_slurm_available
    uid = os.getuid()
    proc = psutil.Process()
    n_process_tree = 1 + len(proc.children(recursive=True))

    try:
        all_procs = list(psutil.process_iter(["pid", "uids"]))
        n_system = len(all_procs)
        n_user = sum(
            1 for p in all_procs
            if p.info["uids"] and p.info["uids"].real == uid
        )
    except Exception:
        n_system = -1
        n_user = -1

    n_slurm = None
    if is_slurm_available():
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
        try:
            n_slurm = sum(
                1 for p in all_procs
                if _proc_in_slurm_job(p, slurm_job_id)
            )
        except Exception:
            n_slurm = -1

    return {
        "process_tree": n_process_tree,
        "user": n_user,
        "uid": uid,
        "system": n_system,
        "slurm": n_slurm,
    }


def print_experiment_overview(monitor, n_workers, proc_counts):
    """Print PID / process counts for each level."""
    print(f"\n{'─'*60}")
    print("Experiment overview")
    print(f"{'─'*60}")
    print(f"  CPUs (available):       {_available_cpus()}")
    print(f"  CPUs (total on node):   {os.cpu_count()}")
    print(f"  Burn workers:           {n_workers}")
    print(f"  Active levels:          {getattr(monitor, 'levels', '?')}")
    print(f"  Processes per level (during workload):")
    uid = proc_counts.get("uid", "?")
    print(f"    process (PID tree):   {proc_counts.get('process_tree', '?')}")
    print(f"    user    (uid={uid}): {' ' * max(0, 4 - len(str(uid)))}{proc_counts.get('user', '?')}")
    print(f"    system  (all):        {proc_counts.get('system', '?')}")
    n_slurm = proc_counts.get("slurm")
    if n_slurm is not None:
        print(f"    slurm   (job={os.environ.get('SLURM_JOB_ID', '?')}): "
              f"{n_slurm}")

    # Per-level sample counts from last run
    print(f"  Samples from last run:")
    for level in getattr(monitor, "levels", []):
        df = monitor.data.data.get(level, pd.DataFrame())
        print(f"    {level:>8s}:             {len(df)}")
    print(f"{'─'*60}")


def _proc_in_slurm_job(proc, slurm_job_id):
    """Check if a process belongs to the given SLURM job."""
    if not slurm_job_id:
        return False
    try:
        env = proc.environ()
        return env.get("SLURM_JOB_ID") == slurm_job_id
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        return False


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

def run_single(backend_name, freq_hz, duration_sec, n_workers=None):
    """Run one benchmark: backend × frequency.

    Returns a dict with summary statistics and the raw DataFrame, or
    None if no data was collected.
    """
    interval = 1.0 / freq_hz
    expected_samples = int(duration_sec * freq_hz)

    monitor = BACKENDS[backend_name]()

    # Hard timeout
    deadline = duration_sec + 30
    old_alarm = None
    def _timeout_handler(signum, frame):
        raise TimeoutError(f"Run exceeded hard deadline of {deadline}s")
    if hasattr(signal, "SIGALRM"):
        old_alarm = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(int(deadline))

    workers = None
    stop_event = None
    t_wall_start = time.perf_counter()
    try:
        workers, stop_event = start_workload(n_workers=n_workers)
        print(f"({len(workers)} burn workers) ", end="", flush=True)
        time.sleep(0.5)
        monitor.start(interval=interval)
        t_setup_done = time.perf_counter()

        # --- measurement window ---
        t_start = time.perf_counter()

        # Snapshot process counts while workload is running
        proc_counts = _snapshot_process_counts()

        time.sleep(duration_sec)
        t_end = time.perf_counter()

        # --- teardown ---
        t_teardown_start = time.perf_counter()
        stop_workload(workers, stop_event)
        monitor.stop()
        t_teardown_done = time.perf_counter()
    except TimeoutError as exc:
        t_end = time.perf_counter()
        t_setup_done = t_setup_done if 't_setup_done' in dir() else t_end
        t_teardown_start = time.perf_counter()
        print(f"      ⚠ {exc}")
        if workers and stop_event:
            try:
                stop_workload(workers, stop_event)
            except Exception:
                pass
        try:
            monitor.stop()
        except Exception:
            pass
        t_teardown_done = time.perf_counter()
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            if old_alarm is not None:
                signal.signal(signal.SIGALRM, old_alarm)
        # Always ensure workers are dead
        if workers and stop_event:
            try:
                stop_workload(workers, stop_event)
            except Exception:
                pass

    # --- inter-run cleanup ---
    # The collector may have reniced our process to +19.  Restore to 0
    # so the next run starts with a clean scheduling state.
    try:
        os.nice(-os.nice(0))  # reset to 0
    except (OSError, PermissionError):
        pass

    # Briefly wait so the OS can fully reclaim child resources and
    # avoid leftover scheduling artifacts leaking into the next run.
    time.sleep(1.0)

    # Extract the "process" level data
    if monitor.data is None:
        return None
    df = monitor.data.data.get("process", pd.DataFrame())
    if df.empty:
        return None

    df = df.copy()
    t0 = df["time"].iloc[0]
    df["time_rel"] = df["time"] - t0
    df["inter_arrival"] = df["time"].diff()
    df["hit"] = df["inter_arrival"].le(interval * 1.5)
    df.loc[df.index[0], "hit"] = True

    actual_duration = t_end - t_start
    setup_time = t_setup_done - t_wall_start
    teardown_time = t_teardown_done - t_teardown_start
    total_wall = t_teardown_done - t_wall_start
    n_actual = len(df)

    return {
        "backend": backend_name,
        "freq_hz": freq_hz,
        "interval": interval,
        "duration": actual_duration,
        "expected": expected_samples,
        "actual": n_actual,
        "hit_rate": min(100.0, n_actual / expected_samples * 100) if expected_samples else 0,
        "mean_iat": df["inter_arrival"].mean(),
        "median_iat": df["inter_arrival"].median(),
        "p95_iat": df["inter_arrival"].quantile(0.95),
        "p99_iat": df["inter_arrival"].quantile(0.99),
        "max_iat": df["inter_arrival"].max(),
        "setup_time": setup_time,
        "teardown_time": teardown_time,
        "total_wall": total_wall,
        "proc_counts": proc_counts,
        "df": df,
        "monitor": monitor,
    }


# ---------------------------------------------------------------------------
# Outlier removal (IQR on hit_rate)
# ---------------------------------------------------------------------------

def remove_outliers(rows):
    """Drop runs whose hit_rate is an IQR outlier.  Returns filtered list."""
    if len(rows) < 4:
        return rows
    rates = np.array([r["hit_rate"] for r in rows])
    q1, q3 = np.percentile(rates, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    kept = [r for r in rows if lo <= r["hit_rate"] <= hi]
    n_removed = len(rows) - len(kept)
    if n_removed:
        print(f"      (removed {n_removed} outlier(s) by IQR on hit_rate)")
    return kept if kept else rows  # never discard all


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Monitor benchmark")
    parser.add_argument("--duration", type=int, default=60,
                        help="Workload duration in seconds (default: 60)")
    parser.add_argument("--repeats", type=int, default=10,
                        help="Number of repetitions per configuration "
                             "(default: 10)")
    parser.add_argument("--backends", type=str, default=None,
                        help="Comma-separated list of backends to test "
                             "(default: all)")
    parser.add_argument("--frequencies", type=str, default=None,
                        help="Comma-separated list of frequencies in Hz "
                             "(default: 1,2,4,8,16)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of CPU burn workers "
                             "(default: auto-detect from SLURM / affinity)")
    args = parser.parse_args()

    _print_cpu_diagnostics()

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    backends = (
        args.backends.split(",") if args.backends
        else list(BACKENDS.keys())
    )
    frequencies = (
        [int(f) for f in args.frequencies.split(",")]
        if args.frequencies else FREQUENCIES
    )

    n_repeats = args.repeats
    overview_printed = False
    agg_summaries = []

    for backend_name in backends:
        if backend_name not in BACKENDS:
            print(f"Unknown backend: {backend_name!r}, skipping")
            continue
        print(f"\n{'='*60}")
        print(f"Backend: {backend_name}")
        print(f"{'='*60}")

        for freq in frequencies:
            interval = 1.0 / freq
            expected = int(args.duration * freq)
            print(f"\n  {freq} Hz (interval={interval:.3f}s), "
                  f"expected≈{expected}, repeats={n_repeats}",
                  flush=True)

            run_rows = []
            all_dfs = []
            last_monitor = None
            for rep in range(1, n_repeats + 1):
                print(f"    run {rep}/{n_repeats} …", end=" ", flush=True)
                try:
                    result = run_single(backend_name, freq, args.duration,
                                         n_workers=args.workers)
                except Exception as exc:
                    print(f"FAILED: {exc}")
                    continue
                if result is None:
                    print("no data")
                    continue

                # Clear psutil's internal cache between runs
                psutil.process_iter.cache_clear()

                n = result["actual"]
                pct = result["hit_rate"]
                dur = result["duration"]
                setup = result["setup_time"]
                td = result["teardown_time"]
                wall = result["total_wall"]
                print(f"{n}/{expected} ({pct:.1f}%) "
                      f"[measure={dur:.1f}s, setup={setup:.1f}s, "
                      f"teardown={td:.1f}s, total={wall:.1f}s]")

                all_dfs.append(result["df"])
                last_monitor = result.pop("monitor")
                last_proc_counts = result.pop("proc_counts", {})
                result.pop("df")
                result["rep"] = rep
                run_rows.append(result)

            if not run_rows:
                print("    ⚠ All runs failed, skipping.")
                continue

            # Print experiment overview once (from the last successful run)
            if not overview_printed and last_monitor is not None:
                print_experiment_overview(
                    last_monitor,
                    _available_cpus(),
                    last_proc_counts,
                )
                overview_printed = True

            # Outlier removal
            kept = remove_outliers(run_rows)

            # Save all per-run raw data (use the median-hit-rate run
            # as the representative for per-sample plots)
            rates = [r["hit_rate"] for r in kept]
            median_idx = int(np.argmin(
                np.abs(np.array(rates) - np.median(rates))
            ))
            # Save representative raw data
            rep_df = all_dfs[kept[median_idx]["rep"] - 1]
            tag = f"{backend_name}_{freq}Hz"
            rep_df.to_csv(
                os.path.join(results_dir, f"{tag}.csv"), index=False
            )

            # Aggregate statistics
            metrics = [
                "hit_rate", "mean_iat", "median_iat",
                "p95_iat", "p99_iat", "max_iat", "actual", "duration",
            ]
            agg = {
                "backend": backend_name,
                "freq_hz": freq,
                "interval": interval,
                "expected": expected,
                "n_runs": len(kept),
            }
            for m in metrics:
                vals = np.array([r[m] for r in kept])
                agg[f"{m}_mean"] = np.mean(vals)
                agg[f"{m}_std"] = np.std(vals, ddof=1) if len(vals) > 1 else 0
            agg_summaries.append(agg)

            print(f"    → avg hit_rate: "
                  f"{agg['hit_rate_mean']:.1f}% "
                  f"± {agg['hit_rate_std']:.1f}%  "
                  f"({len(kept)} runs)")

    # ---- Final summary ----
    if agg_summaries:
        summary = pd.DataFrame(agg_summaries)
        summary_path = os.path.join(results_dir, "summary.csv")
        summary.to_csv(summary_path, index=False)

        print(f"\n{'='*60}")
        print("Aggregated Summary (mean ± std)")
        print(f"{'='*60}")
        display_cols = [
            "backend", "freq_hz", "expected", "n_runs",
            "actual_mean", "actual_std",
            "hit_rate_mean", "hit_rate_std",
            "duration_mean",
            "mean_iat_mean", "p95_iat_mean", "max_iat_mean",
        ]
        display_cols = [c for c in display_cols if c in summary.columns]
        print(summary[display_cols].to_string(index=False, float_format="%.3f"))
        print(f"\nResults written to: {results_dir}/")
    else:
        print("No results collected.")


if __name__ == "__main__":
    main()
