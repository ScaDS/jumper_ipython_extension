"""Carry a prefix's state between processes: capture it, refuse what will not survive, put it back.

Everything the dill replay's two generated scripts do lives here rather than in
their source, so it can be tested by calling it instead of by reading rendered
text. The scripts stay thin on purpose: an import, the prefix cells, and calls
into this module.

**A successful dump proves nothing.** Measured with dill 0.4.0: an open ``w+``
file pickles without complaint and *truncates the file on disk* when restored;
``threading.Lock`` and ``Thread`` pickle and come back as something else. Only
sockets and generators refuse honestly. So the namespace is screened by type
*before* the dump, and a binding that cannot survive refuses the whole mode
rather than producing a checkpoint that looks fine.

What cannot be captured at all - environment variables, ``sys.path``, module
options like ``numpy.seterr``, a view's aliasing, and the BLAS/JIT warm-up the
prefix performed - is out of reach of any namespace checkpoint. The cross-mode
baseline check in the orchestrator is the guard against those.
"""
import io
import json
import os
import platform
import random
import sys
import time

import dill

# Libraries whose generator state lives somewhere we cannot read. jax is
# deliberately absent: its keys are ordinary arrays and travel in the checkpoint.
HOPELESS_RNG_MODULES = ("cupy", "tensorflow")

# How deep into containers the unsafe-value screen looks. A handle three levels
# down is as fatal as one at the top; beyond that the walk costs more than it
# finds.
_SCAN_DEPTH = 3
_SCAN_BUDGET = 20_000

_UNSAFE_TYPE_MODULES = frozenset(
    {
        "_thread",
        "threading",
        "socket",
        "ssl",
        "sqlite3",
        "subprocess",
        "multiprocessing",
        "mmap",
        "selectors",
        "asyncio",
    }
)


class CheckpointTooLarge(RuntimeError):
    """The checkpoint passed the configured byte limit and was abandoned."""


class _CappedWriter:
    """A file that refuses to grow past *limit* bytes.

    A prefix holding tens of gigabytes would otherwise fill a shared filesystem
    before anyone could fall back, and on a cluster that is everyone's problem,
    not just this benchmark's. Deliberately not an ``io`` subclass: the pickler
    only ever calls ``write``, and inheriting brings a ``close`` that flushes a
    file this one has already closed.
    """

    def __init__(self, path: str, limit: int):
        self._file = open(path, "wb")
        self._written = 0
        self._limit = limit

    def write(self, data) -> int:
        self._written += len(data)
        if self._written > self._limit:
            raise CheckpointTooLarge(f"the checkpoint passed {self._limit} bytes")
        return self._file.write(data)

    def flush(self) -> None:
        if not self._file.closed:
            self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    @property
    def written(self) -> int:
        return self._written


def silent_adapter(where: str):
    """A magic adapter that monitors nothing and draws nothing.

    The prefix is replayed through the same hooks the full replay uses - a
    captured ``%perfmonitor_start`` becomes ``magic_adapter.perfmonitor_start(...)``
    - so the checkpoint process needs an adapter even though it measures nothing.
    """
    from jumper_extension.adapters.ai_reviewer.benchmark.measure import build_silent_adapter

    return build_silent_adapter(where)


def _unsafe_reason(value) -> str | None:
    """Why *value* cannot be checkpointed, or None when it can."""
    kind = type(value)
    if kind.__module__.split(".")[0] in _UNSAFE_TYPE_MODULES:
        return f"{kind.__module__}.{kind.__qualname__}"
    if isinstance(value, io.IOBase):
        # In-memory buffers restore faithfully; anything backed by a descriptor
        # comes back reopened, at offset zero, and for a write mode, emptied.
        try:
            value.fileno()
        except Exception:
            return None
        return f"an open file ({getattr(value, 'name', 'unnamed')})"
    return None


def unsafe_bindings(namespace: dict) -> list[str]:
    """Names in *namespace* holding something a checkpoint would silently ruin.

    Reported as ``name: reason`` so the fallback message can say which variable
    stopped the mode instead of leaving the user to guess.
    """
    found: list[str] = []
    budget = _SCAN_BUDGET
    seen: set[int] = set()

    def walk(value, path: str, depth: int) -> None:
        nonlocal budget
        if budget <= 0 or depth > _SCAN_DEPTH:
            return
        budget -= 1
        marker = id(value)
        if marker in seen:
            return
        seen.add(marker)

        reason = _unsafe_reason(value)
        if reason is not None:
            found.append(f"{path}: {reason}")
            return
        if isinstance(value, dict):
            for key, item in list(value.items())[:1000]:
                walk(item, f"{path}[{key!r}]", depth + 1)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for index, item in enumerate(list(value)[:1000]):
                walk(item, f"{path}[{index}]", depth + 1)

    for name, value in list(namespace.items()):
        if name.startswith("__") or name in ("magic_adapter",):
            continue
        walk(value, name, 0)
    return found


def hopeless_rng_modules() -> list[str]:
    """Imported libraries whose RNG state cannot travel in the checkpoint."""
    return [name for name in HOPELESS_RNG_MODULES if name in sys.modules]


def capture_rng() -> dict:
    """The global generator states a restore has to put back.

    Under the full replay the prefix re-seeds on every measurement; a checkpoint
    does not, and a module-level generator comes back seeded from OS entropy. Two
    measurements would then compute different numbers and the result comparison
    would report DIFFERS for a variant that is perfectly correct.
    """
    state: dict = {"random": random.getstate()}
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        state["numpy"] = numpy.random.get_state()
    torch = sys.modules.get("torch")
    if torch is not None:
        state["torch_cpu"] = torch.get_rng_state()
        try:
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                state["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    return state


def restore_rng(path: str) -> list[str]:
    """Put the captured generator states back; returns which ones were restored."""
    if not os.path.exists(path):
        return []
    with open(path, "rb") as handle:
        state = dill.load(handle)

    restored = []
    if "random" in state:
        random.setstate(state["random"])
        restored.append("random")
    numpy = sys.modules.get("numpy")
    if numpy is not None and "numpy" in state:
        numpy.random.set_state(state["numpy"])
        restored.append("numpy")
    torch = sys.modules.get("torch")
    if torch is not None and "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
        restored.append("torch_cpu")
        if "torch_cuda" in state:
            try:
                torch.cuda.set_rng_state_all(state["torch_cuda"])
                restored.append("torch_cuda")
            except Exception:
                pass
    return restored


def environment() -> dict:
    """What the checkpoint was written by, asserted before it is read back.

    Inside one benchmark this cannot drift - same interpreter, same session - but
    a checkpoint is a file, and the day one is reused across runs this is the
    difference between a clear refusal and a restore that quietly returns the
    wrong objects.
    """
    return {
        "python": sys.version.split()[0],
        "dill": dill.__version__,
        "platform": platform.platform(),
    }


def environment_mismatch(meta: dict) -> str:
    """How the running interpreter differs from the one that wrote the checkpoint."""
    written = meta.get("environment") or {}
    current = environment()
    differences = [
        f"{key}: checkpoint {written.get(key)!r} vs current {value!r}"
        for key, value in current.items()
        if written.get(key) != value
    ]
    return "; ".join(differences)


def write_phase(path: str, name: str) -> None:
    """Record how far a restore got, atomically.

    A measurement that fails has to be attributed: before the cell starts, the
    failure is the checkpoint's and the whole benchmark restarts on the full
    replay; from the cell onwards it belongs to the suggestion and goes to the
    repair loop. Guessing wrong sends the model to fix code that was never wrong.
    """
    temporary = f"{path}.part"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(name)
    os.replace(temporary, path)


def read_phase(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def checkpoint(
    namespace: dict,
    paths: dict,
    max_bytes: int,
    prefix_s: float,
    helper_names: tuple = ("magic_adapter",),
) -> None:
    """Screen the prefix's namespace and write the checkpoint, or say why not.

    Refusals are written to the meta file and exit cleanly: the strategy reads
    the reason and falls back to the full replay, which is a normal outcome, not
    an error worth a traceback.
    """
    refused = ""
    unsafe = unsafe_bindings(namespace)
    hopeless = hopeless_rng_modules()
    if unsafe:
        refused = (
            "the prefix left state a checkpoint would silently change: "
            + ", ".join(unsafe[:5])
        )
    elif hopeless:
        refused = (
            f"{', '.join(hopeless)} is imported and its random-number state "
            "cannot be carried into a restored process"
        )

    if refused:
        _write_meta(paths["meta"], {"refused": refused})
        return

    with open(paths["rng"], "wb") as handle:
        dill.dump(capture_rng(), handle)

    # A prefix that carried `%perfmonitor_start` really did start a monitor here,
    # with a collector process behind it. Nothing measures this run, and an
    # orphaned collector would outlive it.
    adapter = namespace.get("magic_adapter")
    if adapter is not None:
        try:
            adapter.perfmonitor_stop("")
        except Exception:
            pass

    for name in helper_names:
        namespace.pop(name, None)

    temporary = f"{paths['checkpoint']}.part"
    writer = _CappedWriter(temporary, max_bytes)
    try:
        dill.dump_module(writer)
        writer.flush()
    except CheckpointTooLarge as error:
        writer.close()
        _remove(temporary)
        _write_meta(paths["meta"], {"refused": str(error)})
        return
    finally:
        writer.close()

    os.replace(temporary, paths["checkpoint"])
    _write_meta(
        paths["meta"],
        {
            "prefix_s": round(prefix_s, 4),
            "size_bytes": os.path.getsize(paths["checkpoint"]),
            "environment": environment(),
        },
    )


def restore(paths: dict) -> float:
    """Load the checkpoint into this process; returns how long it took.

    Raises when the environment does not match what wrote it - the caller has not
    reached the cell yet, so the phase file still says ``loading`` and the failure
    is attributed to the checkpoint rather than to the code under test.
    """
    write_phase(paths["phase"], "loading")
    with open(paths["meta"], encoding="utf-8") as handle:
        meta = json.load(handle)
    mismatch = environment_mismatch(meta)
    if mismatch:
        raise RuntimeError(f"the checkpoint was written elsewhere ({mismatch})")

    started = time.perf_counter()
    dill.load_module(paths["checkpoint"])
    return time.perf_counter() - started


def report_restore(path: str, restore_s: float) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"restore_s": round(restore_s, 4)}, handle)


def read_meta(path: str) -> dict:
    """The checkpoint's metadata, or an empty dict when it is missing or broken."""
    try:
        with open(path, encoding="utf-8") as handle:
            meta = json.load(handle)
    except (OSError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def _write_meta(path: str, payload: dict) -> None:
    temporary = f"{path}.part"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    os.replace(temporary, path)


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
