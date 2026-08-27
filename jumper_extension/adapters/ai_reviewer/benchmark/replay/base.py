"""How the state behind a timed cell is rebuilt, behind one small interface.

Timing a variant means putting the cell back into the state its predecessors
built. *How* that state is reconstructed is a separate question from how the
numbers are read back afterwards, and this is the seam between the two: every
strategy here must leave a JUmPER session export and a fingerprint JSON where
the runner expects them, and beyond that is free to rebuild state however it
can - by replaying the prefix from scratch, by forking a process that already
ran it, or by restoring a checkpoint.

A strategy is orthogonal to the cell's language. ``LanguageAdapter`` decides how
one replay is *written*; a strategy decides how many times the prefix behind it
has to be paid for. Strategies that only serve some languages say so through
``languages``, and the registry falls back to the full replay for the rest -
never by pretending it could rebuild state it cannot.
"""
import os
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from jumper_extension.adapters.ai_reviewer.benchmark.models import FAILED, OK, TIMEOUT
from jumper_extension.adapters.ai_reviewer.language import LanguageAdapter

# The replay modes ``ai.benchmark.replay.mode`` accepts. FULL always works and
# is the fallback for every other mode; the rest are registered by their own
# module once available, so an unbuilt mode degrades instead of failing.
FULL = "full"
FORK = "fork"
DILL = "dill"

_ERROR_TAIL_CHARS = 4000


class StrategyChanged(RuntimeError):
    """Raised when a strategy gave out mid-benchmark and was swapped for another.

    Not an error to report - the swap already happened and the full replay can
    finish the job. It unwinds the run instead, because what was measured before
    the swap and what would be measured after it come from two different
    instruments, and a benchmark's output is a ratio between its measurements.
    Mixing them silently is how a variant ends up divided by a baseline that was
    never timed the same way.
    """


@dataclass
class ReplayContext:
    """What every strategy needs to rebuild the state a cell under test expects."""
    prefix_cells: list[dict]
    interval: float
    work_dir: str
    adapter: LanguageAdapter


@dataclass
class PrepareOutcome:
    """Whether a strategy managed to set itself up, and why not when it did not.

    A false *ok* is not an error: it asks the runner to fall back on the full
    replay, with *reason* explaining the swap to the user.
    """
    ok: bool
    reason: str = ""


@dataclass
class ReplayResult:
    """One replay attempt, before its session export is turned into numbers.

    Only paths and the wall time live here: reading a session back is identical
    for every strategy, so it stays in the runner rather than being reimplemented
    per strategy.
    """
    status: str
    session_path: str = ""
    fingerprint_path: str = ""
    wall_s: float = 0.0
    error: str = ""
    # The strategy itself broke, not the code it was given. Without this a dead
    # zygote reads as a failing suggestion, and every variant gets handed to the
    # repair loop to fix code that was never wrong.
    strategy_broken: bool = False
    # The prefix cell the failure came from, when it came from one. A replay that
    # died before reaching the cell under test has not measured it and has said
    # nothing about it, and a benchmark that reports otherwise sends the repair
    # loop after code that never ran.
    prefix_cell: int | None = None

    @property
    def ok(self) -> bool:
        return self.status == OK


class ReplayStrategy(ABC):
    """One way of getting a cell under test back into its predecessors' state.

    Concrete strategies set ``name`` (the ``ai.benchmark.replay.mode`` value that
    selects them) and, when they cannot serve every language, ``languages``.
    """
    name: str = ""
    # Languages this strategy can serve; empty means every language.
    languages: frozenset[str] = frozenset()

    def __init__(self, context: ReplayContext):
        self.context = context

    def supports(self, adapter: LanguageAdapter) -> bool:
        return not self.languages or adapter.language.lower() in self.languages

    @property
    def target_cell_index(self) -> int:
        """Row the cell under test lands on in the exported history.

        A strategy that replays the prefix puts the target after it; one that
        restores state instead may place it anywhere, so the runner asks rather
        than assuming.
        """
        return len(self.context.prefix_cells)

    def prepare(self) -> PrepareOutcome:
        """One-time setup, done lazily before the first replay.

        The default does nothing, which is right for any strategy that carries no
        state between replays.
        """
        return PrepareOutcome(True)

    @abstractmethod
    def replay(self, code: str, tag: str, timeout: float | None) -> ReplayResult:
        """Run the prefix state plus *code* once, leaving a session and fingerprints."""

    def close(self):
        """Release whatever ``prepare`` acquired. Must be safe to call twice."""

    def child_env(self) -> dict:
        """Keep each replay's own logs beside its session export.

        A replay is a fresh interpreter, so it opens a log directory of its own;
        left alone, a single benchmark would scatter a dozen of them across the
        user's home. Here they land next to the script and zip they describe,
        which is where anyone debugging a failed replay would look.
        """
        return {**os.environ, "JUMPER_LOG_DIR": self.context.work_dir}


def tail(text: str) -> str:
    """The end of a child's stderr - the part that names what actually broke."""
    text = (text or "").strip()
    return text[-_ERROR_TAIL_CHARS:]


def run_script_replay(
    command: list,
    work_dir: str,
    env: dict,
    timeout: float | None,
    session_path: str,
    fingerprint_path: str,
    stale_paths: tuple = (),
) -> ReplayResult:
    """Run one replay script as a child and read its verdict off the filesystem.

    Shared by every strategy that measures in a separate interpreter - which is
    all of them except the fork mode, whose child is reached through a
    supervisor. Leftovers are removed first: measurement tags repeat across
    repair attempts, so a stale session or report from a previous attempt would
    otherwise be read as this one's.
    """
    for stale in (session_path, fingerprint_path, *stale_paths):
        if stale and os.path.exists(stale):
            os.remove(stale)

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=work_dir,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return ReplayResult(
            status=TIMEOUT,
            error=f"Exceeded the {timeout:.0f}s budget and was killed.",
        )
    wall = time.perf_counter() - started

    if completed.returncode != 0:
        # Imported here rather than at module scope: `script` renders cells and
        # this module is what a rendered cell's runtime leans on.
        from jumper_extension.adapters.ai_reviewer.benchmark.script import failing_prefix_cell

        stderr = tail(completed.stderr)
        return ReplayResult(
            status=FAILED,
            error=stderr,
            wall_s=round(wall, 4),
            prefix_cell=failing_prefix_cell(str(command[-1]), stderr),
        )
    if not os.path.exists(session_path):
        return ReplayResult(
            status=FAILED,
            error=f"The run produced no session export.\n{tail(completed.stderr)}",
            wall_s=round(wall, 4),
        )
    return ReplayResult(
        status=OK,
        session_path=session_path,
        fingerprint_path=fingerprint_path,
        wall_s=round(wall, 4),
    )
