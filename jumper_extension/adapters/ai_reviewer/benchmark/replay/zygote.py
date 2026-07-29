"""Run the prefix once, then fork a child per measurement.

The prefix is the same for every run of a benchmark, so paying for it on each of
the ``(1 + variants) x runs`` measurements is the single largest cost the
benchmark has. This process pays it once: it executes the prefix, keeps the
resulting state resident, and forks a child per measurement. A child inherits
that state whole - every object, every import, every JIT-compiled function - and
dies with its mutations, so isolation is preserved for free.

The measurement layout mirrors ``harness.py``: this process owns perfmonitor and
samples itself plus its children, the child brackets the target cell with epoch
timestamps, and the marks are mapped onto the sampler's clock afterwards. What
differs is only where the state comes from.

Three things make the fork safe to read back:

* **The protocol has an fd of its own.** Responses go to a pipe passed in as
  ``--response-fd``, never to stdout: the extension logs a line to stdout the
  moment it is imported, long before any code here could claim the stream, and
  a prefix cell that prints would corrupt it just as easily.
* **The child exits through ``os._exit``**, skipping atexit handlers and the
  interpreter teardown that a forked copy has no business running.
* **A forked child does not always run at the parent's speed**, and both ways it
  can differ are silent. ``probe_fork`` measures each of them across a real fork
  and refuses the mode rather than reporting timings from a machine the notebook
  never ran on.

The GPU is the other refusal: a CUDA context cannot be used after a fork, so a
prefix that initialized one rules this mode out entirely.
"""
import argparse
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

from jumper_extension.adapters.ai_reviewer.benchmark import fingerprint
from jumper_extension.adapters.ai_reviewer.benchmark.harness import (
    clock_offset,
    synthesize_history,
)
from jumper_extension.adapters.ai_reviewer.benchmark.models import FAILED, OK, TIMEOUT
from jumper_extension.core.service import build_perfmonitor_magic_adapter

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
# matmul is too compute-bound to notice. 64MB is past every cache, so the number
# is bandwidth and nothing else.
_PROBE_BYTES = 64 * 1024 * 1024


def _best_matmul(matrix) -> float:
    """Fastest of several identical matmuls, after one discarded warm-up.

    The warm-up matters on the child's side: the first parallel section after a
    fork is where a pool gets rebuilt, and that one-off cost is not what this is
    trying to measure. The minimum, not the mean, because the question is what
    the machine can do, not what it happened to do while something else ran.
    """
    matrix @ matrix
    best = float("inf")
    for _ in range(_PROBE_REPEATS):
        started = time.perf_counter()
        matrix @ matrix
        best = min(best, time.perf_counter() - started)
    return best


def _probe_matrix(numpy):
    """A matmul big enough to be worth timing, and how long the parent takes on it.

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


_PAGE = 4096


def prefault(maps_path: str = "/proc/self/maps") -> int:
    """Resolve this child's inherited pages before anything is timed.

    A fork leaves the child's copy-on-write pages unresolved, and the first read
    of each one costs far more than a later read: measured here, a child summing
    an inherited 160MB array took 13.0ms against its parent's 6.8ms, and 7.1ms
    once the pages had been walked first. That penalty scales with how much of
    the prefix's data a cell touches, so it falls hardest on the fast vectorized
    rewrites a review is looking for - understating their speedup by more than
    half. Walking the pages here moves the cost outside the measured window.

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
    pages it allocated; measured on one machine here, an inherited 160MB array
    summed at 13.3ms in a child against 7.1ms in its parent, while the same child
    summed its *own* fresh array at 6.8ms.

    Both arms run in a single forked child against data the parent made, which is
    the situation every measurement is in.
    """
    try:
        import numpy
    except Exception:
        return {"ok": True, "detail": "numpy is absent, so there is nothing to compare"}

    matrix, _ = _probe_matrix(numpy)
    # arange, never zeros: a calloc'd array is served from the shared zero page
    # until something writes to it, so reading it touches no real memory at all
    # and the arm reports a flat 1.00x no matter what the machine does.
    array = numpy.arange(_PROBE_BYTES // 8, dtype=numpy.float64)
    subjects = {"compute": matrix, "memory": array}
    parent = {name: measure(subjects[name]) for name, measure, _, _ in _ARMS}

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
            with open(path, "w") as handle:
                json.dump(result, handle)
        except BaseException:
            pass
        os._exit(0)
    os.waitpid(pid, 0)

    if not os.path.exists(path):
        return {"ok": False, "detail": "the probe child died before it could report"}
    with open(path) as handle:
        child = json.load(handle)

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

    def __init__(
        self,
        prefix_script: str,
        interval: float,
        prefix_count: int,
        language: str,
        work_dir: str,
        response_fd: int,
    ):
        self.prefix_script = prefix_script
        self.interval = interval
        self.prefix_count = prefix_count
        self.language = language
        self.work_dir = work_dir
        self.adapter = build_perfmonitor_magic_adapter(
            plots_disabled=True,
            plots_disabled_reason="Plotting disabled in the benchmark zygote.",
            display_disabled=True,
            display_disabled_reason="Display disabled in the benchmark zygote.",
        )
        # The prefix runs as __main__ against the adapter the magics were
        # rewritten to call, exactly as the full replay's script does.
        self.namespace = {"__name__": "__main__", "magic_adapter": self.adapter}
        self._protocol = os.fdopen(response_fd, "w")

    def respond(self, payload: dict):
        self._protocol.write(json.dumps(payload) + "\n")
        # Flushed on every response, so the buffer is empty at fork time and no
        # child can inherit half a message and write it out a second time.
        self._protocol.flush()

    def start(self) -> dict:
        """Run the prefix, then report whether this machine may be forked at all."""
        try:
            source = Path(self.prefix_script).read_text(encoding="utf-8")
            exec(compile(source, self.prefix_script, "exec"), self.namespace)
        except BaseException:
            # User code, not the strategy: the caller falls back to the full
            # replay, which will reach the same error through its own path and
            # report it as the failed baseline it is.
            return {"ok": False, "reason": f"the prefix did not run:\n{traceback.format_exc()}"}

        blocker = gpu_blocker()
        if blocker:
            return {"ok": False, "reason": blocker}

        probe = probe_fork(self.work_dir)
        if not probe["ok"]:
            return {"ok": False, "reason": probe["detail"]}
        return {"ok": True, "detail": probe["detail"]}

    def serve(self):
        """Answer replay requests until stdin closes or a stop arrives."""
        while True:
            line = sys.stdin.readline()
            if not line:
                return
            line = line.strip()
            if not line:
                continue
            request = json.loads(line)
            if request.get("cmd") == "stop":
                return
            self.respond(self.replay(request))

    def replay(self, request: dict) -> dict:
        """Fork one child, time it under the monitor, and export its session."""
        code = Path(request["code_path"]).read_text(encoding="utf-8")
        markers_path = request["markers_path"]
        error_path = request["error_path"]
        for stale in (markers_path, error_path):
            _unlink(stale)

        self.adapter.perfmonitor_start(str(self.interval))
        offset = clock_offset()
        started = time.perf_counter()
        try:
            pid = os.fork()
            if pid == 0:
                self._run_child(code, request)  # never returns
            status, error = self._await(pid, request.get("timeout"), error_path)
        finally:
            self.adapter.perfmonitor_stop("")
        wall = round(time.perf_counter() - started, 4)

        if status != OK:
            return {"status": status, "error": error, "wall_s": wall}
        if not os.path.exists(markers_path):
            return {
                "status": FAILED,
                "error": "The run left no timing marks for the cell under test.",
                "wall_s": wall,
            }

        with open(markers_path) as handle:
            markers = json.load(handle)
        self.adapter.service.cell_history.data = synthesize_history(
            prefix_count=self.prefix_count,
            target_code=code,
            language=self.language,
            start_epoch=float(markers["start"]),
            end_epoch=float(markers["end"]),
            offset=offset,
        )
        self.adapter.export_session(request["session_path"])
        return {"status": OK, "wall_s": wall}

    def _run_child(self, code: str, request: dict):
        """Run the target cell in the inherited state, then leave without cleanup."""
        try:
            self._protocol.close()  # the parent's stream is none of the child's business
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
            with open(request["markers_path"], "w") as handle:
                json.dump({"start": start, "end": end}, handle)
            # After the end mark on purpose: fingerprinting is not the cell's cost.
            fingerprint.dump(request["output_names"], self.namespace, request["fingerprint_path"])
        except BaseException:
            _write(request["error_path"], traceback.format_exc())
            _flush_std()
            os._exit(1)
        _flush_std()
        # _exit, not sys.exit: a forked child must not run atexit handlers or
        # interpreter teardown that belong to the process it was copied from.
        os._exit(0)

    def _await(self, pid: int, timeout: float | None, error_path: str) -> tuple[str, str]:
        """Wait for the child, killing it once its budget is spent."""
        deadline = time.perf_counter() + timeout if timeout else None
        while True:
            done, status = os.waitpid(pid, os.WNOHANG)
            if done:
                break
            if deadline is not None and time.perf_counter() > deadline:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                return TIMEOUT, f"Exceeded the {timeout:.0f}s budget and was killed."
            time.sleep(_POLL_S)

        if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0:
            return OK, ""
        detail = _read(error_path)
        if detail:
            return FAILED, detail
        if os.WIFSIGNALED(status):
            return FAILED, f"The run was killed by signal {os.WTERMSIG(status)}."
        return FAILED, f"The run exited with status {os.WEXITSTATUS(status)}."


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


def _unlink(path: str):
    try:
        os.remove(path)
    except OSError:
        pass


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JUmPER benchmark fork zygote")
    parser.add_argument("--prefix-script", required=True)
    parser.add_argument("--interval", type=float, required=True)
    parser.add_argument("--prefix-count", type=int, required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--language", default="python")
    parser.add_argument(
        "--response-fd",
        type=int,
        required=True,
        help="inherited pipe the parent reads answers from; never stdout",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    zygote = Zygote(
        prefix_script=args.prefix_script,
        interval=args.interval,
        prefix_count=args.prefix_count,
        language=args.language,
        work_dir=args.work_dir,
        response_fd=args.response_fd,
    )
    ready = zygote.start()
    zygote.respond(ready)
    if not ready["ok"]:
        return 0  # the reason is the answer; the caller falls back on its own
    zygote.serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
