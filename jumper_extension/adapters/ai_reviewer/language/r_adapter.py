"""The R half of the benchmark's language-static seams (Phase 5a).

R cells reach the benchmark through the JUmPER wrapper kernel. This adapter
provides the two seams that need no running replay: a syntax gate (R's own
parser, shelled out through ``Rscript``) and best-effort extraction of the names
a cell assigns. Replaying R in a fresh runtime and fingerprinting R values is
Phase 5b; until then the adapter claims no ``RUN`` capability, so the benchmark
degrades to a warning instead of executing R through the Python machinery.
"""
import os
import re
import shutil
import subprocess
import tempfile

from jumper_extension.adapters.ai_reviewer.language.base import (
    VALIDATE_SYNTAX,
    CapabilityNotSupported,
    LanguageAdapter,
    ReplayArtifact,
    ReplayRequest,
    SyntaxResult,
)

_PARSE_TIMEOUT_S = 20.0

# Best-effort static extraction of top-level R bindings (no R needed): left
# assigns `name <- / <<- / = expr`, right assigns `expr -> / ->> name`, and
# `assign("name", ...)`. Enough to fingerprint outputs; 5b may refine via R.
_LEFT = re.compile(r"^\s*([A-Za-z.][\w.]*)\s*(?:<<-|<-|=)(?!=)")
_RIGHT = re.compile(r"(?:->>|->)\s*([A-Za-z.][\w.]*)\s*$")
_ASSIGN_FN = re.compile(r"""\bassign\s*\(\s*["']([^"']+)["']""")


class RAdapter(LanguageAdapter):
    """R cells: syntax gate and assigned-name extraction (no timed run yet)."""
    language = "r"

    def __init__(self, rscript: str | None = None):
        # None means "auto-detect"; an explicit value (including "") is honoured,
        # so callers and tests can force the no-R path.
        self._rscript = shutil.which("Rscript") if rscript is None else rscript
        # Without R there is no parser to gate with, so drop the capability and
        # let the benchmark warn honestly rather than pretend to validate.
        self.caps = frozenset({VALIDATE_SYNTAX}) if self._rscript else frozenset()

    def validate_syntax(self, code: str) -> SyntaxResult:
        if not self._rscript:
            raise CapabilityNotSupported("Rscript not found: cannot validate R syntax")

        handle = tempfile.NamedTemporaryFile(
            "w",
            suffix=".R",
            delete=False,
            encoding="utf-8",
        )
        try:
            handle.write(code)
            handle.close()
            completed = subprocess.run(
                [
                    self._rscript,
                    "-e",
                    "invisible(parse(file=commandArgs(TRUE)[1]))",
                    handle.name,
                ],
                capture_output=True,
                text=True,
                timeout=_PARSE_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return SyntaxResult(ok=False, error="R parser timed out")
        finally:
            _unlink(handle.name)

        if completed.returncode == 0:
            return SyntaxResult(ok=True)
        return SyntaxResult(ok=False, error=_parse_error(completed.stderr))

    def output_names(self, code: str) -> list[str]:
        names: list[str] = []
        for statement in _statements(code):
            left = _LEFT.match(statement)
            if left:
                names.append(left.group(1))
            right = _RIGHT.search(statement)
            if right:
                names.append(right.group(1))
            names.extend(_ASSIGN_FN.findall(statement))
        return list(dict.fromkeys(names))

    def render_replay(self, request: ReplayRequest) -> ReplayArtifact:
        raise CapabilityNotSupported(
            "R replay is not implemented yet (Phase 5b): benchmark cannot run R cells"
        )


def _statements(code: str) -> list[str]:
    """Rough split into top-level statements: drop comments, split on ; and \\n."""
    statements: list[str] = []
    for line in code.splitlines():
        statements.extend(_strip_comment(line).split(";"))
    return statements


def _strip_comment(line: str) -> str:
    """Drop a trailing ``#`` comment, ignoring ``#`` inside quotes (best-effort)."""
    in_single = in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:index]
    return line


def _parse_error(stderr: str) -> str:
    text = (stderr or "").strip()
    return text[-1000:] if text else "R parse error"


def _unlink(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
