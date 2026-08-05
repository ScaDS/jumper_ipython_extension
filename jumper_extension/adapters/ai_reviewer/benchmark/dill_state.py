"""Carry a prefix's state between processes: capture it, refuse what will not survive, put it back.

Everything the dill replay's two generated scripts do lives here rather than in
their source, so it can be tested by calling it instead of by reading rendered
text. The scripts stay thin on purpose: an import, the prefix cells, and calls
into this module.

**A successful dump proves nothing on its own.** Measured with dill 0.4.0: an
open ``w+`` file pickles without complaint and *truncates the file on disk* when
restored; ``threading.Lock`` and ``Thread`` pickle and come back as something
else. Only sockets and generators refuse honestly.

So the boundary is drawn where the whole object graph is walked anyway - inside
the pickler. :func:`install_pickler_guard` gives the unsafe types a reducer that
refuses, which reaches a handle hidden on an attribute, in an object array or ten
containers deep, without this module reimplementing that traversal. The shallow
:func:`unsafe_bindings` walk that runs first exists only to name the offending
variable in the message; the guard is what actually decides.

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
import socket
import sys
import threading
import time
import types

import dill

# The user's state lives in a module of this name, never in ``__main__``.
_STATE_MODULE = "jumper_user_state"

_LOCK_TYPE = type(threading.Lock())
_RLOCK_TYPE = type(threading.RLock())

# Libraries whose generator state lives somewhere we cannot read. jax is
# deliberately absent: its keys are ordinary arrays and travel in the checkpoint.
HOPELESS_RNG_MODULES = ("cupy", "tensorflow")

# How deep into containers the unsafe-value screen looks. A handle three levels
# down is as fatal as one at the top; beyond that the walk costs more than it
# finds.
_SCAN_DEPTH = 3
_SCAN_BUDGET = 20_000

# Bindings the generated scripts injected, by identity - see ``own``.
_OWNED: dict = {}

# The restore's paths, kept here rather than in the namespace a checkpoint
# overwrites.
PATHS: dict = {}

# The checkpoint's metadata, read back by the restore.
META: dict = {}

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


class UnsafeState(RuntimeError):
    """A value was reached that a checkpoint would silently change."""


def _refusing_reducer(description):
    def refuse(pickler, obj):
        raise UnsafeState(description(obj))

    return refuse


def install_pickler_guard() -> None:
    """Make the pickler refuse the values a restore would ruin.

    Called only in the checkpoint process - it mutates dill's dispatch table, and
    the notebook's own use of dill is none of our business.

    Registration is by exact type, which is enough: anything *holding* one of
    these is reached through it. A ``threading.Event`` is not listed, for
    instance, because the pickler walks into the lock it owns.
    """
    for kind in (
        io.TextIOWrapper,
        io.BufferedReader,
        io.BufferedWriter,
        io.BufferedRandom,
        io.FileIO,
    ):
        dill.register(kind)(
            _refusing_reducer(lambda obj: f"an open file ({getattr(obj, 'name', 'unnamed')})")
        )
    dill.register(_LOCK_TYPE)(_refusing_reducer(lambda obj: "a lock"))
    dill.register(_RLOCK_TYPE)(_refusing_reducer(lambda obj: "a reentrant lock"))
    dill.register(threading.Thread)(_refusing_reducer(lambda obj: f"a thread ({obj.name})"))
    dill.register(socket.socket)(_refusing_reducer(lambda obj: "a socket"))


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
    """A magic adapter that never draws or prints, and monitors only if asked.

    The prefix runs through the same hooks the full replay uses, and a captured
    ``%perfmonitor_start`` renders to ``magic_adapter.perfmonitor_start(...)`` -
    so this adapter really can end up monitoring, whatever the checkpoint process
    intends. That is why the checkpoint stops it again before dumping.
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

    **Advisory, not the boundary.** This walk is shallow and budgeted, and it is
    here only so the refusal can name the variable rather than leaving the user
    to guess. What actually decides is :func:`install_pickler_guard`, which
    refuses from inside the pickler and therefore misses nothing.
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
        else:
            # A handle is as likely to sit on an object as in a list.
            attributes = getattr(value, "__dict__", None)
            if isinstance(attributes, dict):
                for key, item in list(attributes.items())[:1000]:
                    walk(item, f"{path}.{key}", depth + 1)

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
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            # Deliberately unguarded: a device generator we cannot read is a
            # reproducibility hole, and refusing the mode beats hiding it.
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(path: str, expected: list | None = None) -> list[str]:
    """Put the captured generator states back, or refuse to pretend they are back.

    A missing artifact used to read as "nothing to restore", which is the same
    silence it exists to prevent: every measurement would reseed from entropy and
    a correct variant would be reported as computing something else.
    """
    if not os.path.exists(path):
        raise UnsafeState(f"the captured random-number state is missing ({path})")
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
            torch.cuda.set_rng_state_all(state["torch_cuda"])
            restored.append("torch_cuda")

    missing = sorted(set(expected or []) - set(restored))
    if missing:
        raise UnsafeState(
            f"the checkpoint captured {', '.join(missing)} but this process could not "
            "restore it, so its measurements would not be reproducible"
        )
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


def new_state():
    """A module of the user's own, kept apart from the script that drives it.

    Prefix and target code run in here, and this is what gets checkpointed - so
    the bookkeeping the generated scripts need lives in ``__main__`` where the
    user's names can never reach it, and vice versa. The one name this namespace
    carries that the user did not write is ``magic_adapter``, which the hooks
    need and which is dropped again before the dump.
    """
    state = types.ModuleType(_STATE_MODULE)
    sys.modules[_STATE_MODULE] = state
    return state


def attach_adapter(state, where: str):
    """Give *state* the adapter its cells' magics will call."""
    adapter = silent_adapter(where)
    state.magic_adapter = adapter
    _OWNED["magic_adapter"] = id(adapter)
    return adapter


def run_cell(state, raw_cell: str, cell_magics: list | None = None, index=None) -> None:
    """Execute one cell in *state*, through the hooks a live notebook uses."""
    from jumper_extension.adapters.script_writer import is_pure_magic_cell, transform_cell_code

    magics = list(cell_magics or [])
    adapter = state.magic_adapter
    adapter.on_pre_run_cell(raw_cell, magics, is_pure_magic_cell(raw_cell))
    source = transform_cell_code(raw_cell, magics)
    exec(compile(source, f"<cell {index}>" if index is not None else "<cell under test>", "exec"),
         state.__dict__)
    adapter.on_post_run_cell("")


def use_paths(paths: dict) -> dict:
    """Keep the restore's own paths in this module, out of the user's namespace."""
    PATHS.clear()
    PATHS.update(paths)
    return PATHS


def _drop_owned(namespace: dict) -> None:
    for name, marker in list(_OWNED.items()):
        if name in namespace and id(namespace[name]) == marker:
            del namespace[name]


def checkpoint(
    state,
    paths: dict,
    max_bytes: int,
    prefix_s: float,
) -> None:
    """Screen the prefix's namespace and write the checkpoint, or say why not.

    Refusals are written to the meta file and exit cleanly: the strategy reads
    the reason and falls back to the full replay, which is a normal outcome, not
    an error worth a traceback.
    """
    hopeless = hopeless_rng_modules()
    if hopeless:
        _write_meta(
            paths["meta"],
            {
                "refused": (
                    f"{', '.join(hopeless)} is imported and its random-number state "
                    "cannot be carried into a restored process"
                )
            },
        )
        return

    namespace = state.__dict__
    # Named early so a refusal can point at a variable; the pickler guard below
    # is what actually decides, and it sees everything this walk cannot.
    named = unsafe_bindings(namespace)

    try:
        streams = capture_rng()
    except Exception as error:
        _write_meta(
            paths["meta"],
            {"refused": f"the prefix's random-number state could not be captured: {error}"},
        )
        return
    with open(paths["rng"], "wb") as handle:
        dill.dump(streams, handle)

    # A prefix that carried `%perfmonitor_start` really did start a monitor here,
    # with a collector process behind it. Nothing measures this run, and an
    # orphaned collector would outlive it.
    adapter = namespace.get("magic_adapter")
    if adapter is not None:
        try:
            adapter.perfmonitor_stop("")
        except Exception:
            pass

    _drop_owned(namespace)

    install_pickler_guard()
    temporary = f"{paths['checkpoint']}.part"
    writer = _CappedWriter(temporary, max_bytes)
    try:
        dill.dump_module(writer, module=state)
        writer.flush()
    except (CheckpointTooLarge, UnsafeState) as error:
        writer.close()
        _remove(temporary)
        _remove(paths["rng"])
        # The scan usually knows which variable it was; the guard always knows
        # what it was, even when the value was buried where the scan cannot look.
        refusal = (
            "the prefix left state a checkpoint would silently change: "
            + ", ".join(named[:3])
            if named
            else str(error)
        )
        _write_meta(paths["meta"], {"refused": refusal})
        return
    finally:
        writer.close()

    os.replace(temporary, paths["checkpoint"])
    _write_meta(
        paths["meta"],
        {
            "prefix_s": round(prefix_s, 4),
            "size_bytes": os.path.getsize(paths["checkpoint"]),
            "rng": sorted(streams),
            "environment": environment(),
        },
    )


def restore(paths: dict):
    """Load the checkpoint into a namespace of its own; returns ``(state, seconds)``.

    Raises when the environment does not match what wrote it, or when the random
    state the checkpoint captured cannot be put back - in both cases the cell has
    not started, so the phase file still says ``loading`` and the failure is the
    checkpoint's rather than the code's.
    """
    write_phase(paths["phase"], "loading")
    meta = read_meta(paths["meta"])
    if not meta:
        raise UnsafeState(f"the checkpoint metadata is missing or unreadable ({paths['meta']})")
    mismatch = environment_mismatch(meta)
    if mismatch:
        raise UnsafeState(f"the checkpoint was written elsewhere ({mismatch})")

    if meta.get("rng") and not os.path.exists(paths["rng"]):
        raise UnsafeState(f"the captured random-number state is missing ({paths['rng']})")

    state = new_state()
    started = time.perf_counter()
    dill.load_module(paths["checkpoint"], module=state)
    elapsed = time.perf_counter() - started
    META.clear()
    META.update(meta)
    return state, elapsed


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
