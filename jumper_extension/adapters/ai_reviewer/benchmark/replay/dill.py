"""Measure on top of a restored checkpoint instead of a replayed prefix.

The prefix runs once, into a `dill` checkpoint; every measurement is a fresh
interpreter that loads it. That buys what the fork mode buys - the prefix paid
once - without any of what the fork mode costs: no inherited pages to fault in,
no thread pool lost to `fork()`, and an RSS that describes one process rather
than a zygote and a child holding the same data.

It is not free of distortion, only of *that* distortion. A checkpoint carries
values, not the process around them: environment variables, `sys.path`, module
options like `numpy.seterr`, a view's aliasing and every BLAS or JIT cache the
prefix warmed are gone. The last one is the mirror of the fork mode's: where a
forked child pays for memory it inherited, a restored one pays for a first call
the prefix should already have paid. Neither shows up from inside the mode,
because the baseline is restored the same way as the variants and agrees with
them - which is why the cross-mode baseline check exists and is on by default.

What the mode refuses, it refuses before writing anything - and not by trusting
the dump, which proves nothing on its own (an open file pickles happily and comes
back truncated). `dill_state` gives the unsafe types a reducer that raises, so
the refusal happens inside the pickler, wherever in the object graph the value
is hiding.
"""
import logging
import os
import shutil
import subprocess
import sys
import tempfile

from jumper_extension.adapters.ai_reviewer.benchmark import dill_state
from jumper_extension.adapters.ai_reviewer.benchmark.replay.base import (
    DILL,
    PrepareOutcome,
    ReplayResult,
    ReplayStrategy,
    run_script_replay,
    tail,
)
from jumper_extension.adapters.ai_reviewer.benchmark.script import (
    build_checkpoint_script,
    build_restore_script,
    failing_checkpoint_cell,
)
from jumper_extension.config.loader import load_config

logger = logging.getLogger("extension")

# A prefix is allowed to be slow - it is the user's own notebook - but not
# endless: without a deadline a hung prefix would take the benchmark with it and
# print nothing, and falling back to the full replay is always available.
_PREPARE_TIMEOUT_S = 1800.0

# When a restore costs nearly as much as the prefix it replaces, the mode is
# doing work for nothing. Not a refusal: the user asked for it, so this only
# says so, once.
_NO_GAIN_SHARE = 0.8

# A prefix this cheap costs nothing to replay either way, so which mode runs it
# is not worth a warning - and at these durations the two figures are noise.
_MIN_PREFIX_S = 0.5

# Phases a restore reports. Only while the phase is ``cell_started`` is the code
# under test actually running; before that the checkpoint is being restored, and
# after it this benchmark is fingerprinting and exporting. A failure in either is
# ours, and blaming it on a suggestion would send the model off to fix code that
# did nothing wrong.
_CELL_PHASES = ("cell_started",)


class DillReplayStrategy(ReplayStrategy):
    """Rebuild the state once into a checkpoint, restore it per measurement."""
    name = DILL
    languages = frozenset({"python"})

    def __init__(self, context):
        super().__init__(context)
        self._state_dir = ""
        self._paths: dict = {}
        self._prefix_s = 0.0
        self._warned_no_gain = False

    @property
    def target_cell_index(self) -> int:
        """The cell under test is the whole history: no prefix was replayed."""
        return 0

    def prepare(self) -> PrepareOutcome:
        # A directory of our own making, with a name nobody else can predict: the
        # work_dir may be the caller's, and close() deletes this tree whole.
        state_dir = tempfile.mkdtemp(prefix="dill_state-", dir=self.context.work_dir)
        os.chmod(state_dir, 0o700)
        self._state_dir = state_dir
        self._paths = {
            "checkpoint": os.path.join(state_dir, "checkpoint.pkl"),
            "meta": os.path.join(state_dir, "meta.json"),
            "rng": os.path.join(state_dir, "rng.pkl"),
        }

        script = build_checkpoint_script(
            self.context.prefix_cells,
            self._paths,
            _max_checkpoint_bytes(self.context.work_dir),
            os.path.join(state_dir, "checkpoint.py"),
        )
        log_path = os.path.join(state_dir, "checkpoint.log")
        try:
            # Straight to a file, never a pipe: nobody drains the child while a
            # chatty prefix runs, and a full pipe would wedge it.
            with open(log_path, "w") as log:
                completed = subprocess.run(
                    [sys.executable, script],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=self.context.work_dir,
                    env=self.child_env(),
                    timeout=_PREPARE_TIMEOUT_S,
                )
        except subprocess.TimeoutExpired:
            return PrepareOutcome(
                False,
                f"the prefix did not finish within {_PREPARE_TIMEOUT_S:.0f}s",
            )
        except OSError as error:
            return PrepareOutcome(False, f"the checkpoint process could not start: {error}")

        if completed.returncode != 0:
            log = tail(_read_tail(log_path))
            cell = failing_checkpoint_cell(log)
            where = f" at cell {cell}" if cell is not None else ""
            return PrepareOutcome(
                False,
                f"the prefix failed while being checkpointed{where}.\n{log}",
            )

        meta = dill_state.read_meta(self._paths["meta"])
        if meta.get("refused"):
            return PrepareOutcome(False, meta["refused"])
        if not os.path.exists(self._paths["checkpoint"]):
            return PrepareOutcome(
                False,
                f"no checkpoint was written.\n{tail(_read_tail(log_path))}",
            )

        self._prefix_s = float(meta.get("prefix_s") or 0.0)
        logger.debug(
            f"[JUmPER]: benchmark dill checkpoint: {meta.get('size_bytes', 0)} bytes "
            f"for a {self._prefix_s}s prefix"
        )
        return PrepareOutcome(True)

    def replay(self, code: str, tag: str, timeout: float | None) -> ReplayResult:
        work_dir = self.context.work_dir
        session_path = os.path.join(work_dir, f"{tag}_session.zip")
        fingerprint_path = os.path.join(work_dir, f"{tag}_fingerprint.json")
        phase_path = os.path.join(self._state_dir, f"{tag}.phase")
        restore_report = os.path.join(self._state_dir, f"{tag}.restore.json")

        script = build_restore_script(
            {**self._paths, "phase": phase_path},
            target_code=code,
            interval=self.context.interval,
            # A timed run always fingerprints its outputs: verification rides
            # along with the replay and is cheap beside it.
            fingerprint_names=self.context.adapter.output_names(code),
            session_path=session_path,
            fingerprint_path=fingerprint_path,
            restore_report_path=restore_report,
            output_path=os.path.join(work_dir, f"{tag}_restore.py"),
        )

        result = run_script_replay(
            [sys.executable, script],
            work_dir=work_dir,
            env=self.child_env(),
            timeout=timeout,
            session_path=session_path,
            fingerprint_path=fingerprint_path,
            # Tags repeat across repair attempts; yesterday's phase file would
            # otherwise vouch for today's failure.
            stale_paths=(phase_path, restore_report),
        )
        if not result.ok:
            return self._blame(result, phase_path)
        self._warn_if_no_gain(restore_report)
        return result

    def close(self):
        if not self._state_dir:
            return
        shutil.rmtree(self._state_dir, ignore_errors=True)
        self._state_dir = ""

    def _blame(self, result: ReplayResult, phase_path: str) -> ReplayResult:
        """Decide whether a failed measurement is the checkpoint's fault or the cell's.

        A restore that never reached the cell is not a failing suggestion, and
        must not be handed to the repair loop to fix code that never ran.
        """
        phase = dill_state.read_phase(phase_path)
        if phase in _CELL_PHASES:
            return result
        stage = {
            "": "the restore never started",
            "loading": "the checkpoint could not be restored",
            "setup": "the monitor could not be started on the restored state",
            "cell_finished": "the cell ran, but its results could not be captured",
            "exporting": "the cell ran, but its session could not be exported",
            "completed": "the run finished and then failed anyway",
        }.get(phase, f"the restore stopped at {phase!r}")
        return ReplayResult(
            status=result.status,
            wall_s=result.wall_s,
            strategy_broken=True,
            error=f"{stage}. {result.error}",
        )

    def _warn_if_no_gain(self, restore_report: str):
        """Say once when restoring costs about what running the prefix cost.

        dill keeps modules by reference, so a restore re-imports everything the
        prefix imported. A prefix that is mostly `import torch` therefore gets
        traded for a restore that is mostly `import torch`, and the mode is
        paying disk for nothing.
        """
        if self._warned_no_gain or self._prefix_s < _MIN_PREFIX_S:
            return
        report = dill_state.read_meta(restore_report)
        restore_s = float(report.get("restore_s") or 0.0)
        if not restore_s or restore_s < _NO_GAIN_SHARE * self._prefix_s:
            return

        self._warned_no_gain = True
        logger.warning(
            f"[JUmPER]: restoring the dill checkpoint takes {restore_s}s against the "
            f"{self._prefix_s}s the prefix itself took, so this mode is saving little: "
            "a restore re-imports every module the prefix imported. The two figures "
            "exclude what both modes pay anyway (interpreter start, monitoring) and "
            "the one-off cost of writing the checkpoint. Consider --replay-mode full."
        )


def _max_checkpoint_bytes(work_dir: str) -> int:
    """The configured ceiling, or what the filesystem can actually spare.

    The cap alone does not protect a disk with less free space than the cap, so
    the smaller of the two wins - and a margin is left, because a filesystem that
    reaches zero takes more than this benchmark down with it.
    """
    replay = load_config().ai.benchmark.replay
    configured = int(float(replay.dill_max_checkpoint_gb) * 1024 ** 3)
    try:
        free = shutil.disk_usage(work_dir).free
    except OSError:
        return configured
    return max(min(configured, int(free * 0.9)), 0)


def _read_tail(path: str, limit: int = 8192) -> str:
    """The end of a log, however the child chose to encode it.

    Read as bytes and decoded with replacement: a prefix that printed one invalid
    byte used to turn a clean fallback into a UnicodeDecodeError, and a chatty one
    would have had its whole log pulled into memory to show the last lines of it.
    """
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(handle.tell() - limit, 0))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
