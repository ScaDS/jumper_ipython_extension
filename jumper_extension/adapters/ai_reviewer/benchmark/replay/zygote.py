"""Hold the prefix state, and hand out forked copies of it on request.

The prefix is the same for every run of a benchmark, so paying for it on each of
the ``(1 + variants) x runs`` measurements is the single largest cost the
benchmark has. This process pays it once: it executes the prefix, keeps the
resulting state resident, and forks a child per measurement. A child inherits
that state whole - every object, every import, every JIT-compiled function - and
dies with its mutations, so no measurement can leak into the next.

**Nothing here monitors anything.** That is the point of the process boundary
above it: a monitor runs a sampler thread, and only the forking thread survives
a ``fork()``, so a lock held by a vanished thread would stay locked forever in
the child. The supervisor owns the monitor and the session export; this process
keeps as few threads as the prefix leaves it with, and does the forking.

What still lives here is what only the holder of the state can do:

* run the prefix, once, against an adapter whose magics are inert - a prefix
  cell calling ``%perfmonitor_start`` must not start a sampler in the very
  process this design exists to keep thread-free;
* decide whether this machine may be forked at all (``probe_fork``,
  ``gpu_blocker``);
* fork, wait, and kill on timeout;
* in the child: pay the fork's own costs before the clock starts - walking the
  inherited pages and rebuilding the thread pool - then run the cell, write
  epoch marks, count what the kernel had to do anyway, dump output fingerprints,
  and leave through ``os._exit`` without running teardown that belongs to the
  process it was copied from.

Each of those preamble steps exists because a cost that belongs to *being a
forked child* was landing inside the window meant to measure the cell, and none
of them average out: every measurement is a fresh child, so there is no such
thing as a warmed-up run. What cannot be moved out - a reference count written
into an inherited page is a real copy - is counted and reported instead.
"""
import argparse
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path

from jumper_extension.adapters.ai_reviewer.benchmark import fingerprint
from jumper_extension.adapters.ai_reviewer.benchmark.models import FAILED, OK, TIMEOUT
from jumper_extension.adapters.ai_reviewer.benchmark.replay.lifetime import (
    die_with_parent,
    kill_group,
    own_process_group,
)
from jumper_extension.adapters.ai_reviewer.benchmark.replay.link import JsonLink

# How often the parent checks on a running child. Only the wall time of a
# measurement is affected; the cell itself is timed by its own epoch marks.
_POLL_S = 0.002

# Big enough that a BLAS call engages a thread pool at all, which is the only
# thing the probe now needs of it - the verdict is a thread count, so the size no
# longer has to be calibrated against measurement noise.
_PROBE_SIZE = 1200
# Read across a fork purely as a diagnostic: 64MB is past most last-level caches,
# though not the largest server ones.
_PROBE_BYTES = 64 * 1024 * 1024

_PAGE = 4096

# madvise(2): populate the page tables for a range as if it had been read,
# without copying anything. Linux 5.14 and later; older kernels refuse it and the
# walk falls back to touching the pages itself.
_MADV_POPULATE_READ = 22

# Small enough to cost under a millisecond, large enough that BLAS takes its
# parallel path rather than the serial shortcut it keeps for tiny matrices.
_WARMUP_SIZE = 64


class InertMagicAdapter:
    """Accepts the magic calls a replayed prefix makes, and does nothing.

    The prefix runs here only to build state. Monitoring, reporting and session
    export belong to the supervisor, and a prefix cell calling
    ``%perfmonitor_start`` must not start a sampler thread in the process that
    is about to fork.

    Unknown names still raise, so a cell using a magic the real adapter does not
    have fails here exactly as it would in a full replay - parity between the
    modes matters more than swallowing the error would be convenient.
    """

    def __getattr__(self, name: str):
        from jumper_extension.core.service import PerfmonitorMagicAdapter

        if not hasattr(PerfmonitorMagicAdapter, name):
            raise AttributeError(name)
        return _accept_anything


def _accept_anything(*args, **kwargs):
    return None


def resident_regions(smaps_path: str = "/proc/self/smaps"):
    """Anonymous private ranges that actually hold something, largest first.

    ``smaps`` rather than ``maps`` because it carries each region's resident
    size, and address space is a poor guide to what is there: measured on one
    zygote, 2544MB of regions against 370MB actually in memory - the rest an
    untouched reservation of the kind ``numpy.empty`` leaves behind. Walking by
    region would spend real time on all of it and gain nothing, because reads of
    an untouched page are served from the shared zero page.

    File-backed regions are left out deliberately. They carry the same
    first-touch cost - an inherited 200MB ``numpy.memmap`` summed at 20.8ms in a
    child against 11.9ms in its parent - but walking them was measured *not* to
    help (19.5ms before, 22.4ms after, at twice the walk cost), so what they cost
    is reported through the page-fault warning instead of paid for here.
    """
    start = end = 0
    for line in _lines(smaps_path):
        fields = line.split()
        # A header line ("7f..-7f.. rw-p 00000000 00:00 0 [path]") starts each
        # region and is followed by its statistics, one per line.
        if len(fields) >= 5 and "-" in fields[0] and len(fields[1]) == 4:
            start = end = 0
            perms = fields[1]
            path = fields[5] if len(fields) > 5 else ""
            # Where the prefix's arrays and objects live. The kernel's own
            # ([vdso], [vvar], ...) are not ours to walk.
            if "r" in perms and "p" in perms and (not path or path == "[heap]"):
                try:
                    start, end = (int(bound, 16) for bound in fields[0].split("-"))
                except ValueError:
                    start = end = 0
        elif start and fields and fields[0] == "Rss:":
            resident = int(fields[1]) if len(fields) > 1 and fields[1].isdigit() else 0
            if resident:
                yield start, end
            start = end = 0


def prefault(smaps_path: str = "/proc/self/smaps") -> int:
    """Resolve this child's inherited pages before anything is timed.

    A fork leaves the child's copy-on-write pages unresolved, and the first read
    of each one costs far more than a later read: measured on one machine, a
    child summing an inherited 160MB array took 13.0ms against its parent's
    6.8ms, and 7.1ms once the pages had been walked first. The penalty scales
    with how much of the prefix's data a cell touches, so it falls hardest on the
    fast vectorized rewrites a review is looking for. Walking the pages here
    moves the cost outside the measured window.

    The kernel is asked to do the walk (``MADV_POPULATE_READ``) rather than
    touching the addresses ourselves. That is faster, needs no numpy, and - the
    reason it matters most - it cannot crash: reading raw addresses from a
    snapshot of the region list means a region freed in between takes the process
    down with a signal Python cannot catch, and a dead zygote is reported as the
    measured suggestion failing. An older kernel refuses the call, and only then
    is the strided read used instead.

    Two things it does not cover, both measured rather than assumed:

    * **Writes.** Reading a Python object *writes* to the page it lives on,
      because the reference count lives inside the object, and a write to an
      inherited page is a real copy rather than bookkeeping. Walking ahead cannot
      pre-pay that - copying is exactly what a fork avoids - so what remains is
      counted per measurement and reported instead. Measured: an inherited
      20-million-element list summed in 737ms against 107ms on a second pass.
    * **File-backed mappings**, for the reason given in ``resident_regions``.

    Linux only: elsewhere there is no ``smaps`` to read, this returns 0, and the
    penalty stays inside the measurement.

    Returns the pages populated.
    """
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
    except Exception:
        return 0

    touched = 0
    for start, end in resident_regions(smaps_path):
        size = end - start
        if libc.madvise(
            ctypes.c_void_p(start),
            ctypes.c_size_t(size),
            ctypes.c_int(_MADV_POPULATE_READ),
        ) == 0 or _read_through(ctypes, start, size):
            touched += size // _PAGE
    return touched


def _read_through(ctypes, start: int, size: int) -> bool:
    """Touch one byte per page, for kernels without ``MADV_POPULATE_READ``.

    One strided pass rather than a Python loop over pages: the work is the
    kernel's either way, and the loop tripled it.
    """
    try:
        import numpy

        buffer = (ctypes.c_ubyte * size).from_address(start)
        int(numpy.frombuffer(buffer, dtype=numpy.uint8)[::_PAGE].sum())
        return True
    except Exception:
        return False


def _lines(path: str) -> list[str]:
    try:
        with open(path) as handle:
            return handle.readlines()
    except OSError:
        return []


def warm_thread_pool() -> bool:
    """Rebuild the thread pool the fork destroyed, before anything is timed.

    Only the forking thread survives a ``fork()``, so a BLAS or OpenMP pool the
    prefix had built is gone in the child. The library notices and builds a new
    one - but lazily, on the next parallel operation, which in this arrangement
    is the measured cell itself. Measured here: a matmul taking 105ms in the
    parent took 132ms as the first one in a child, and 115ms when a 0.8ms warm-up
    had already paid for the pool.

    It never averages out on its own. Every measurement is a fresh child, so
    dropping the first run as a warm-up does not help - the second and third are
    just as cold - and a fixed cost of tens of milliseconds is a rounding error
    against a slow baseline and the whole of a fast rewrite.

    **Limitation: this warms BLAS, and only BLAS.** That is what numpy operations
    go through, which covers most of what gets benchmarked, but a cell whose
    parallelism comes from elsewhere - OpenMP inside numba or scikit-learn,
    torch's own inter-op pools - still builds that pool inside the measured
    window. Covering them would mean a per-library list that is always one
    library out of date; what such a case leaves behind is a raised compute ratio
    in the probe's logged timings.
    """
    try:
        import numpy

        block = numpy.ones((_WARMUP_SIZE, _WARMUP_SIZE))
        block @ block
        return True
    except Exception:
        return False


def task_count() -> int:
    """How many OS threads this process has right now."""
    try:
        return len(os.listdir("/proc/self/task"))
    except OSError:
        return 1


def probe_fork(work_dir: str) -> dict:
    """Can a forked child still use the cores the prefix was using?

    Asked structurally, not by stopwatch. Only the forking thread survives a
    ``fork()``, so a BLAS or OpenMP pool built during the prefix is gone in the
    child; whether it comes *back* is what matters, because a cell timed on one
    core instead of twelve produces plausible, badly wrong numbers. Counting the
    threads the child reaches after its own parallel region answers that with no
    measurement noise at all - observed here: a parent on 12 threads forks a
    child that starts on 1 and returns to 12, while the same test under
    ``OPENBLAS_NUM_THREADS=1`` reads 1 against 1 and correctly sees nothing lost.

    An earlier version of this decided by comparing timings across the fork, and
    refused a healthy machine in 6 runs out of 20: two unreplicated measurements
    of a few milliseconds cannot separate a real effect from a busy machine. The
    timings are still taken - they are the only way an *unknown* slowdown would
    ever show up - but they are reported for the log rather than used to refuse.

    **Limitations.** The thread question is asked through numpy, so without it
    nothing is checked at all and the mode is allowed on trust. Pools that are
    not BLAS - OpenMP inside numba or scikit-learn, torch's inter-op pools - are
    neither warmed nor counted here. And the answer is about the machine, while
    the distortions it looks for scale with what a *cell* touches: that half is
    the per-measurement page-fault warning's job, not this one's.
    """
    try:
        import numpy
    except Exception:
        return {"ok": True, "detail": "numpy is absent, so there is no thread pool to lose"}

    generator = numpy.random.default_rng(0)
    matrix = generator.random((_PROBE_SIZE, _PROBE_SIZE))
    # arange, never zeros: a calloc'd array is served from the shared zero page
    # until something writes to it, so reading it would touch no real memory.
    array = numpy.arange(_PROBE_BYTES // 8, dtype=numpy.float64)

    matrix @ matrix  # build the pool here, so the child faces a real one at fork
    parent = {
        "threads": task_count(),
        "compute": _timed(lambda: matrix @ matrix),
        "memory": _timed(array.sum),
    }

    path = os.path.join(work_dir, "fork_probe.json")
    _unlink(path)
    pid = os.fork()
    if pid == 0:
        try:
            # Exactly what a measurement does, so the probe sees the path that
            # ships rather than an idealised one.
            started = time.perf_counter()
            pages = prefault()
            walk_s = time.perf_counter() - started
            # Kept out of walk_s, which prices page faults, but done before the
            # arms so the probe measures the preamble a measurement really gets.
            warm_thread_pool()
            child = {
                "compute": _timed(lambda: matrix @ matrix),
                "memory": _timed(array.sum),
                "walk_s": walk_s,
                "walk_pages": pages,
            }
            # After the matmul: a pool is rebuilt lazily, on first use.
            child["threads"] = task_count()
            _write(path, json.dumps(child))
        except BaseException:
            pass
        os._exit(0)
    os.waitpid(pid, 0)

    child = _read_json(path)
    if child is None:
        return {"ok": False, "detail": "the probe child died before it could report"}

    # Losing more than half the cores is a timing error of more than 2x, and the
    # failure this guards against is total - 12 threads down to 1, not 12 to 11.
    if parent["threads"] > 1 and child["threads"] * 2 < parent["threads"]:
        return {
            "ok": False,
            "detail": (
                f"a forked child gets back only {child['threads']} of the "
                f"{parent['threads']} threads its parent was using, so every "
                "measurement would be timed on fewer cores than the notebook used"
            ),
        }

    return {
        "ok": True,
        "threads": f"{child['threads']}/{parent['threads']}",
        "timings": {
            name: round(child[name] / parent[name], 2) for name in ("compute", "memory")
        },
        # What the cheapest kind of fault costs on this machine, measured by the
        # walk that just did several hundred thousand of them. Deliberately the
        # cheap kind: it makes every per-measurement estimate a floor, and a
        # floor can be stated without ever exceeding the thing it describes.
        "fault_cost_s": child["walk_s"] / child["walk_pages"] if child["walk_pages"] else 0.0,
        "walk_pages": child["walk_pages"],
        "walk_s": round(child["walk_s"], 3),
    }


def _timed(action) -> float:
    started = time.perf_counter()
    action()
    return time.perf_counter() - started


def gpu_blocker() -> str:
    """Why forking would be unsafe on this GPU state, or '' when it is not.

    A CUDA context does not survive a fork: the child gets an initialization
    error on its first call. Only modules the prefix actually imported are
    inspected, so a notebook that never went near an accelerator is not held back
    by one being installed.

    **This is a best-effort list, and it errs in both directions.** Torch is
    asked whether its CUDA context is actually initialized; cupy and jax are not,
    so importing either refuses the mode even when the work is on the CPU. In the
    other direction it does not know about TensorFlow, numba.cuda, PyCUDA, ROCm
    or OpenCL, nor about ``mpi4py`` once MPI is initialized, where forking is
    undefined - so a prefix using those can still be forked and should not be.
    """
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_initialized():
                return "torch holds an initialized CUDA context, which no forked child can use"
        except Exception:
            pass
    for name in ("cupy", "jax"):
        if name in sys.modules:
            return f"{name} is imported and may hold a GPU context, which no forked child can use"
    return ""


class Zygote:
    """Holds the prefix state and hands out forked copies of it."""

    def __init__(self, prefix_script: str, work_dir: str, link: JsonLink):
        self.prefix_script = prefix_script
        self.work_dir = work_dir
        self.link = link
        # The prefix runs as __main__ against the adapter its magics were
        # rewritten to call, exactly as the full replay's script does - except
        # that here the adapter does nothing.
        self.namespace = {"__name__": "__main__", "magic_adapter": InertMagicAdapter()}

    def start(self) -> dict:
        """Run the prefix, then report whether this machine may be forked at all."""
        try:
            source = Path(self.prefix_script).read_text(encoding="utf-8")
            exec(compile(source, self.prefix_script, "exec"), self.namespace)
        except BaseException:
            # User code, not the strategy: the caller falls back to the full
            # replay, which reaches the same error through its own path and
            # reports it as the failed baseline it is.
            return {"ok": False, "reason": f"the prefix did not run:\n{traceback.format_exc()}"}

        blocker = gpu_blocker()
        if blocker:
            return {"ok": False, "reason": blocker}

        probe = probe_fork(self.work_dir)
        if not probe["ok"]:
            return {"ok": False, "reason": probe["detail"]}
        # Passed through whole: what the probe measured is what the caller needs
        # to state a page-fault warning in this machine's own units.
        return probe

    def serve(self):
        """Fork a child per request until the supervisor stops asking."""
        for request in self.link.requests_until_closed():
            self.link.answer(self.fork_once(request))

    def fork_once(self, request: dict) -> dict:
        """Fork one child to run the cell, and say how it went.

        No timing beyond the kill budget happens here: the child writes its own
        epoch marks, and the supervisor - which owns the sampler clock - is what
        turns them into a measurement.
        """
        code = Path(request["code_path"]).read_text(encoding="utf-8")
        error_path = request["error_path"]
        for stale in (request["markers_path"], error_path):
            _unlink(stale)

        pid = os.fork()
        if pid == 0:
            self._run_child(code, request)  # never returns
        # Both sides place the child in its own group; whichever runs first wins
        # and the loser's failure is meaningless. Without it a timeout could only
        # kill the cell, leaving whatever the cell started behind.
        own_process_group(pid)
        outcome = self._await(pid, request.get("timeout"), error_path)
        if outcome["status"] != OK:
            return outcome
        # What the cell cost the kernel, carried up beside what it cost the clock:
        # only the caller knows the machine's price per fault, so the judgement of
        # whether it mattered is made there.
        marks = _read_json(request["markers_path"]) or {}
        return {
            **outcome,
            "faults": marks.get("faults", 0),
            "duration_s": float(marks.get("end", 0.0)) - float(marks.get("start", 0.0)),
        }

    def _run_child(self, code: str, request: dict):
        """Run the target cell in the inherited state, then leave without cleanup."""
        # Its own group, so a timeout reaches whatever the cell starts; and dead
        # when the zygote is, so an abandoned measurement cannot keep a full copy
        # of the prefix alive with nobody waiting for it.
        own_process_group()
        die_with_parent()
        try:
            self.link.close()  # the supervisor's stream is none of the child's business
        except BaseException:
            pass
        try:
            handle = os.open(
                request["stdout_path"],
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                0o644,
            )
            os.dup2(handle, 1)
            os.dup2(handle, 2)
            # stdin is the zygote's request channel: a cell calling input() would
            # otherwise eat the next measurement request off it.
            os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
            os.close(handle)
        except BaseException:
            pass

        try:
            compiled = compile(code, "<cell under test>", "exec")
            # Before the start mark: resolving inherited pages is the fork's
            # cost, not the cell's, and leaving it inside the window understates
            # exactly the fastest variants.
            # Both before the start mark: resolving inherited pages and
            # rebuilding a thread pool are the fork's costs, not the cell's, and
            # inside the window they understate exactly the fastest variants.
            prefault()
            warm_thread_pool()
            # Page faults bracket the cell for the same reason the clocks do.
            # A fault here is the kernel making inherited memory usable, and for
            # Python objects that means copying the page a reference count was
            # written to - work the notebook never did, charged to the cell.
            before = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
            start = time.time()
            exec(compiled, self.namespace)
            end = time.time()
            faults = resource.getrusage(resource.RUSAGE_SELF).ru_minflt - before
            _write(
                request["markers_path"],
                json.dumps({"start": start, "end": end, "faults": faults}),
            )
            # After the end mark on purpose: fingerprinting is not the cell's cost.
            fingerprint.dump(
                request["output_names"],
                self.namespace,
                request["fingerprint_path"],
            )
        except BaseException:
            _write(request["error_path"], traceback.format_exc())
            _flush_std()
            os._exit(1)
        _flush_std()
        # _exit, not sys.exit: a forked child must not run atexit handlers or
        # interpreter teardown that belong to the process it was copied from.
        os._exit(0)

    def _await(self, pid: int, timeout: float | None, error_path: str) -> dict:
        """Wait for the child, killing it once its budget is spent."""
        deadline = time.perf_counter() + timeout if timeout else None
        while True:
            done, status = os.waitpid(pid, os.WNOHANG)
            if done:
                break
            if deadline is not None and time.perf_counter() > deadline:
                kill_group(pid)
                os.waitpid(pid, 0)
                return {
                    "status": TIMEOUT,
                    "error": f"Exceeded the {timeout:.0f}s budget and was killed.",
                }
            time.sleep(_POLL_S)

        if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0:
            return {"status": OK}
        detail = _read(error_path)
        if detail:
            return {"status": FAILED, "error": detail}
        if os.WIFSIGNALED(status):
            return {"status": FAILED, "error": f"The run was killed by signal {os.WTERMSIG(status)}."}
        return {"status": FAILED, "error": f"The run exited with status {os.WEXITSTATUS(status)}."}


def _flush_std():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except BaseException:
            pass


def _write(path: str, text: str):
    try:
        with open(path, "w") as handle:
            handle.write(text)
    except BaseException:
        pass


def _read(path: str) -> str:
    try:
        with open(path) as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _read_json(path: str) -> dict | None:
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _unlink(path: str):
    try:
        os.remove(path)
    except OSError:
        pass


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JUmPER benchmark fork zygote")
    parser.add_argument("--prefix-script", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument(
        "--response-fd",
        type=int,
        required=True,
        help="inherited pipe the supervisor reads answers from; never stdout",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not die_with_parent():
        # Either the kernel will not promise it, or the supervisor is already
        # gone - and running the prefix for a supervisor that no longer exists
        # would hold a full copy of it for nothing.
        if os.getppid() == 1:
            return 0
    zygote = Zygote(
        prefix_script=args.prefix_script,
        work_dir=args.work_dir,
        link=JsonLink(sys.stdin, args.response_fd),
    )
    ready = zygote.start()
    zygote.link.answer(ready)
    if not ready["ok"]:
        return 0  # the reason is the answer; the caller falls back on its own
    zygote.serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
