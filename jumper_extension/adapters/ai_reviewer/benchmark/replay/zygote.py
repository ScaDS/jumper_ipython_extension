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
* in the child: walk the inherited pages, run the cell, write epoch marks and
  output fingerprints, and leave through ``os._exit`` without running teardown
  that belongs to the process it was copied from.
"""
import argparse
import json
import os
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

# The compute arm has to run long enough that rebuilding a thread pool and
# scheduling it are noise beside it. Measured at 600x600 the two sides differ by
# 4x on an untouched OpenBLAS purely from jitter, and at 13ms they still flapped
# past a 1.25 margin under load. Sizes are tried in order until one is slow
# enough to mean something.
_PROBE_SIZES = (600, 1200, 2400, 4000)
_PROBE_MIN_S = 0.05
_PROBE_REPEATS = 5
# The memory arm streams an array the child inherited rather than one it made:
# reading copy-on-write memory can cost far more than reading your own, and a
# matmul is too compute-bound to notice. 64MB is past most last-level caches,
# though not the largest server ones.
_PROBE_BYTES = 64 * 1024 * 1024

_PAGE = 4096


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


def _best_matmul(matrix) -> float:
    """Fastest of several identical matmuls, after one discarded warm-up.

    The minimum, not the mean, because the question is what the machine can do,
    not what it happened to do while something else ran.
    """
    matrix @ matrix
    best = float("inf")
    for _ in range(_PROBE_REPEATS):
        started = time.perf_counter()
        matrix @ matrix
        best = min(best, time.perf_counter() - started)
    return best


def _probe_matrix(numpy) -> tuple:
    """A matmul big enough to be worth timing, and the parent's time on it.

    A dedicated Generator, never the global one: the zygote's RNG state is
    inherited by every child, and perturbing it here would make a fork's results
    differ from a full replay's for no reason at all.
    """
    generator = numpy.random.default_rng(0)
    matrix = None
    parent = 0.0
    for size in _PROBE_SIZES:
        matrix = generator.random((size, size))
        parent = _best_matmul(matrix)
        if parent >= _PROBE_MIN_S:
            break
    return matrix, parent


def _first_stream(array) -> float:
    """One full read of *array*, with nothing discarded.

    Deliberately not a best-of: the cost this arm exists to catch is paid on the
    *first* touch of an inherited page, so a warm-up would hide exactly what is
    being looked for. This is what a measured cell actually experiences.
    """
    started = time.perf_counter()
    array.sum()
    return time.perf_counter() - started


def prefault(maps_path: str = "/proc/self/maps") -> int:
    """Resolve this child's inherited pages before anything is timed.

    A fork leaves the child's copy-on-write pages unresolved, and the first read
    of each one costs far more than a later read: measured on one machine, a
    child summing an inherited 160MB array took 13.0ms against its parent's
    6.8ms, and 7.1ms once the pages had been walked first. The penalty scales
    with how much of the prefix's data a cell touches, so it falls hardest on the
    fast vectorized rewrites a review is looking for. Walking the pages here
    moves the cost outside the measured window.

    One strided pass per region rather than a Python loop over pages: the work is
    the kernel's either way, and the loop tripled it. Returns the pages touched.
    """
    try:
        import ctypes

        import numpy
    except Exception:
        # Without numpy there is no strided read to do this cheaply - and a
        # prefix that never imported it has little inherited data to speak of.
        return 0

    touched = 0
    try:
        with open(maps_path) as handle:
            regions = handle.readlines()
    except OSError:
        return 0

    for line in regions:
        fields = line.split()
        if len(fields) < 2:
            continue
        perms = fields[1]
        path = fields[5] if len(fields) > 5 else ""
        # Anonymous private memory only: that is where the prefix's arrays and
        # objects live. File-backed regions come from the page cache already, and
        # the kernel's own ([vdso], [vvar], ...) are not ours to walk.
        if "r" not in perms or "p" not in perms:
            continue
        if path and path != "[heap]":
            continue
        try:
            start, end = (int(bound, 16) for bound in fields[0].split("-"))
            size = end - start
            buffer = (ctypes.c_ubyte * size).from_address(start)
            int(numpy.frombuffer(buffer, dtype=numpy.uint8)[::_PAGE].sum())
        except Exception:
            continue
        touched += size // _PAGE
    return touched


# Each arm's margin is set against the size of the failure it exists to catch,
# not to a single house number. Both err on the side of refusing: a probe that
# wrongly refuses costs only speed, while one that wrongly passes ships numbers
# nobody can tell are wrong.
_ARMS = (
    (
        "compute",
        _best_matmul,
        # Losing a 12-core pool costs 4-8x, so the margin can stay well clear of
        # the jitter a shared machine adds without ever missing the real thing.
        1.5,
        "it lost the thread pool the prefix built, and every measurement would be "
        "timed on fewer cores than the notebook used",
    ),
    (
        "memory",
        _first_stream,
        # The unwalked-page penalty measured here was 1.9x, and this arm runs
        # after the same prefault a measurement does, so anything past a third
        # means the walk did not take.
        1.3,
        "the first read of an inherited page still costs more than a later one "
        "even after the pages are walked, so the penalty lands inside the "
        "measured window in proportion to how much of the prefix's data a cell "
        "touches - which understates exactly the vectorized rewrites this review "
        "exists to find",
    ),
)


def probe_fork(work_dir: str) -> dict:
    """Does a forked child run at the speed of the process that forked it?

    Two ways it can fail to, both silent, and neither visible to a check that
    compares results rather than times. *compute* catches a thread pool that did
    not survive the fork - only the forking thread does, and a BLAS pool rebuilt
    at one thread produces plausible, badly wrong numbers. *memory* catches the
    slower path a child can be put on when it reads pages it inherited instead of
    pages it allocated.

    Both arms run in a single forked child against data the parent made, which is
    the situation every measurement is in.
    """
    try:
        import numpy
    except Exception:
        return {"ok": True, "detail": "numpy is absent, so there is nothing to compare"}

    matrix, compute_parent = _probe_matrix(numpy)
    # arange, never zeros: a calloc'd array is served from the shared zero page
    # until something writes to it, so reading it touches no real memory at all
    # and the arm reports a flat 1.00x no matter what the machine does.
    array = numpy.arange(_PROBE_BYTES // 8, dtype=numpy.float64)
    subjects = {"compute": matrix, "memory": array}
    parent = {"compute": compute_parent, "memory": _first_stream(array)}

    path = os.path.join(work_dir, "fork_probe.json")
    _unlink(path)
    pid = os.fork()
    if pid == 0:
        try:
            # Exactly what a measurement does, so the probe validates the path
            # that ships rather than an idealised one.
            started = time.perf_counter()
            pages = prefault()
            result = {name: measure(subjects[name]) for name, measure, _, _ in _ARMS}
            result["prefault_s"] = time.perf_counter() - started
            result["prefault_pages"] = pages
            _write(path, json.dumps(result))
        except BaseException:
            pass
        os._exit(0)
    os.waitpid(pid, 0)

    child = _read_json(path)
    if child is None:
        return {"ok": False, "detail": "the probe child died before it could report"}

    seen = []
    for name, _, tolerance, consequence in _ARMS:
        ratio = child[name] / parent[name]
        measured = (
            f"{child[name] * 1000:.0f}ms in a child vs {parent[name] * 1000:.0f}ms "
            "in the parent"
        )
        seen.append(f"{name} {ratio:.2f}x")
        if ratio > tolerance:
            return {
                "ok": False,
                "detail": (
                    f"a forked child is {ratio:.1f}x slower on the {name} probe "
                    f"({measured}): {consequence}"
                ),
            }
    return {
        "ok": True,
        "detail": (
            f"forked child matches its parent ({', '.join(seen)}); "
            f"{child['prefault_pages']} pages walked per measurement in "
            f"{child['prefault_s'] * 1000:.0f}ms"
        ),
    }


def gpu_blocker() -> str:
    """Why forking would be unsafe on this GPU state, or '' when it is not.

    A CUDA context does not survive a fork: the child gets an initialization
    error on its first call. Only modules the prefix actually imported are
    inspected, so a notebook that never touched the GPU is never held back.
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
        return {"ok": True, "detail": probe["detail"]}

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
        return self._await(pid, request.get("timeout"), error_path)

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
            prefault()
            start = time.time()
            exec(compiled, self.namespace)
            end = time.time()
            _write(request["markers_path"], json.dumps({"start": start, "end": end}))
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
