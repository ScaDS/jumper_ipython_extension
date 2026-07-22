"""The language-specific seams of the benchmark, behind one small interface.

The replay and timing machinery is language-agnostic; only three things depend
on the cell's language: whether a suggestion parses, which names it binds (so a
variant can be checked for computing the same thing), and how the replay is
built and launched. Each language provides those through a ``LanguageAdapter``;
everything else - timing, medians, signature comparison - stays shared.

An adapter declares which of these it can actually do through ``caps``, so a
language that cannot, say, verify results is handled by skipping that step with
a reason - never by pretending it passed. See ``registry`` for lookup and the
no-capability fallback.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

# The two benchmark steps an adapter may implement, matching the configurable
# check levels (ai.benchmark.checks). A step whose capability is absent is
# skipped with a warning unless the user turned it off deliberately. RUN covers
# the timed replay and the result fingerprinting together: there is nothing to
# fingerprint without an execution, and capturing it is cheap next to the run.
VALIDATE_SYNTAX = "validate_syntax"
RUN = "run"


class CapabilityNotSupported(RuntimeError):
    """Raised when a seam is used that the adapter never claimed in ``caps``."""


@dataclass
class SyntaxResult:
    """Whether a candidate parses, and why not when it does not."""
    ok: bool
    error: str = ""


@dataclass
class ReplayRequest:
    """Everything needed to render one replay of the prefix plus a target cell.

    ``output_path`` is an extension-less base path for the replay artifact; each
    adapter appends the suffix its runtime expects (``.py``, ``.R``, ...), so the
    runner never has to know the language.
    """
    prefix_cells: list[dict]
    target_code: str
    interval: float
    output_names: list[str]
    session_path: str
    fingerprint_path: str
    output_path: str
    work_dir: str


@dataclass
class ReplayArtifact:
    """A rendered replay: the file to run and the command that runs it.

    Running ``command`` must leave a JUmPER session export at
    ``ReplayRequest.session_path`` and a fingerprint JSON at ``fingerprint_path``;
    the shared runner reads both back the same way for every language.
    """
    script_path: str
    command: list[str]


class LanguageAdapter(ABC):
    """The per-language half of the benchmark.

    Concrete adapters set ``language`` (matched case-insensitively against the
    ``language`` recorded on each cell) and ``caps`` (the subset of
    ``VALIDATE_SYNTAX`` / ``RUN`` they actually implement).
    """
    language: str = ""
    caps: frozenset[str] = frozenset()

    def supports(self, capability: str) -> bool:
        return capability in self.caps

    @abstractmethod
    def validate_syntax(self, code: str) -> SyntaxResult:
        """Cheap parse check, so broken code never costs a replay."""

    @abstractmethod
    def output_names(self, code: str) -> list[str]:
        """Top-level names *code* binds, to fingerprint for result verification."""

    @abstractmethod
    def render_replay(self, request: ReplayRequest) -> ReplayArtifact:
        """Write the replay artifact and say how to launch it."""
