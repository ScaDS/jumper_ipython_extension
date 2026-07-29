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
from abc import ABC, abstractmethod
from dataclasses import dataclass

from jumper_extension.adapters.ai_reviewer.benchmark.models import OK
from jumper_extension.adapters.ai_reviewer.language import LanguageAdapter

# The replay modes ``ai.benchmark.replay.mode`` accepts. FULL always works and
# is the fallback for every other mode; the rest are registered by their own
# module once available, so an unbuilt mode degrades instead of failing.
FULL = "full"
FORK = "fork"
DILL = "dill"

_ERROR_TAIL_CHARS = 4000


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
