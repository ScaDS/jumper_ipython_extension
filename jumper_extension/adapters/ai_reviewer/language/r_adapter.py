"""The R half of the benchmark's language seams (Design B).

R cells reach the benchmark through the JUmPER wrapper kernel. This adapter
provides the three seams that depend on the language: a syntax gate (R's own
parser, shelled out through ``Rscript``), best-effort extraction of the names a
cell assigns, and - under Design B - a replay that runs prefix + target in a
fresh R runtime. The replay does not touch perfmonitor or the session export:
``render_replay`` hands the shared harness an ``Rscript`` command, and the
generated R script only brackets the target with epoch markers and dumps output
fingerprints. Everything language-neutral (profiling, session export, reading
the result back) stays in the harness and the shared runner.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from jumper_extension.adapters.ai_reviewer.language.base import (
    RUN,
    VALIDATE_SYNTAX,
    CapabilityNotSupported,
    LanguageAdapter,
    ReplayArtifact,
    ReplayRequest,
    SyntaxResult,
)
from jumper_extension.adapters.ai_reviewer.language.r_script import build_r_script

_HARNESS_MODULE = "jumper_extension.adapters.ai_reviewer.benchmark.harness"

_PARSE_TIMEOUT_S = 20.0

# Best-effort static extraction of top-level R bindings (no R needed): left
# assigns `name <- / <<- / = expr`, right assigns `expr -> / ->> name`, and
# `assign("name", ...)`. Enough to fingerprint outputs; 5b may refine via R.
_LEFT = re.compile(r"^\s*([A-Za-z.][\w.]*)\s*(?:<<-|<-|=)(?!=)")
_RIGHT = re.compile(r"(?:->>|->)\s*([A-Za-z.][\w.]*)\s*$")
_ASSIGN_FN = re.compile(r"""\bassign\s*\(\s*["']([^"']+)["']""")


class RAdapter(LanguageAdapter):
    """R cells: syntax gate, assigned-name extraction, and a harness-run replay."""
    language = "r"

    def __init__(self, rscript: str | None = None):
        # None means "auto-detect"; an explicit value (including "") is honoured,
        # so callers and tests can force the no-R path.
        self._rscript = shutil.which("Rscript") if rscript is None else rscript
        # Both real seams need Rscript; without it drop every capability and let
        # the benchmark warn honestly rather than pretend it could run R.
        self.caps = (
            frozenset({VALIDATE_SYNTAX, RUN}) if self._rscript else frozenset()
        )

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
            # A `name <- function(...)` binding is a helper, not a data result;
            # it cannot be fingerprinted, so leave it out rather than let its
            # empty signature muddy the verification (see fingerprint.compare_all).
            if left and not statement[left.end():].lstrip().startswith("function"):
                names.append(left.group(1))
            right = _RIGHT.search(statement)
            if right:
                names.append(right.group(1))
            names.extend(_ASSIGN_FN.findall(statement))
        return list(dict.fromkeys(names))

    def render_replay(self, request: ReplayRequest) -> ReplayArtifact:
        if not self._rscript:
            raise CapabilityNotSupported("Rscript not found: cannot run R cells")

        markers_path = f"{request.output_path}.markers.json"
        script_path = build_r_script(
            prefix_cells=request.prefix_cells,
            target_code=request.target_code,
            output_names=request.output_names,
            markers_path=markers_path,
            fingerprint_path=request.fingerprint_path,
            output_path=f"{request.output_path}.R",
        )
        # Kept so the exported session shows the cell under test, matching the
        # Python path; the harness reads it only to fill the target row.
        target_code_file = f"{request.output_path}.target.R"
        Path(target_code_file).write_text(request.target_code, encoding="utf-8")

        command = [
            sys.executable,
            "-m",
            _HARNESS_MODULE,
            "--run",
            json.dumps([self._rscript, script_path]),
            "--session",
            request.session_path,
            "--markers",
            markers_path,
            "--interval",
            str(request.interval),
            "--prefix-count",
            str(len(request.prefix_cells)),
            "--work-dir",
            request.work_dir,
            "--target-code-file",
            target_code_file,
            "--language",
            "r",
        ]
        return ReplayArtifact(script_path=script_path, command=command)


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
