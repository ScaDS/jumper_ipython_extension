"""The Python half of the benchmark: the behaviour that shipped before adapters.

Everything here is exactly what the orchestrator and runner did inline - a
``compile`` syntax gate, ``ast``-based assigned-name extraction, and a replay
script driven through the JUmPER magic hooks and launched with this
interpreter. Pulling it behind the adapter interface changes nothing for
Python; it just makes the same three seams swappable for other languages.
"""
import sys

from jumper_extension.adapters.ai_reviewer.benchmark import fingerprint
from jumper_extension.adapters.ai_reviewer.benchmark.script import build_script
from jumper_extension.adapters.ai_reviewer.language.base import (
    RUN,
    VALIDATE_SYNTAX,
    LanguageAdapter,
    ReplayArtifact,
    ReplayRequest,
    SyntaxResult,
)


class PythonAdapter(LanguageAdapter):
    """Python cells: full support for every benchmark step."""
    language = "python"
    caps = frozenset({VALIDATE_SYNTAX, RUN})

    def validate_syntax(self, code: str) -> SyntaxResult:
        try:
            compile(code, "<suggestion>", "exec")
            return SyntaxResult(ok=True)
        except SyntaxError as error:
            return SyntaxResult(ok=False, error=f"{error.__class__.__name__}: {error}")

    def output_names(self, code: str) -> list[str]:
        return fingerprint.assigned_names(code)

    def render_replay(self, request: ReplayRequest) -> ReplayArtifact:
        script_path = build_script(
            prefix_cells=request.prefix_cells,
            target_code=request.target_code,
            interval=request.interval,
            fingerprint_names=request.output_names,
            session_path=request.session_path,
            fingerprint_path=request.fingerprint_path,
            output_path=f"{request.output_path}.py",
        )
        return ReplayArtifact(
            script_path=script_path,
            command=[sys.executable, script_path],
        )
