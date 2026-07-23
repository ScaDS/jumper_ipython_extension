import shutil

import pytest

from jumper_extension.adapters.ai_reviewer.language import (
    VALIDATE_SYNTAX,
    CapabilityNotSupported,
    ReplayRequest,
    get_adapter,
)
from jumper_extension.adapters.ai_reviewer.language.r_adapter import RAdapter

_HAS_R = shutil.which("Rscript") is not None
_needs_r = pytest.mark.skipif(not _HAS_R, reason="Rscript not installed")


# --- registration / capability ---

def test_r_language_resolves_to_the_r_adapter():
    assert isinstance(get_adapter("r"), RAdapter)
    assert isinstance(get_adapter("R"), RAdapter)


def test_run_is_claimed_when_rscript_present():
    from jumper_extension.adapters.ai_reviewer.language import RUN

    assert RUN in RAdapter(rscript="/usr/bin/Rscript").caps


def test_capability_drops_when_rscript_is_missing():
    from jumper_extension.adapters.ai_reviewer.language import RUN

    adapter = RAdapter(rscript="")

    assert adapter.caps == frozenset()
    assert RUN not in adapter.caps
    with pytest.raises(CapabilityNotSupported):
        adapter.validate_syntax("x <- 1")


@_needs_r
def test_validate_syntax_is_claimed_when_rscript_present():
    assert VALIDATE_SYNTAX in RAdapter().caps


# --- validate_syntax (needs R) ---

@_needs_r
def test_valid_r_passes():
    result = RAdapter().validate_syntax("z <- sqrt(4)\na = 1; b <- 2")

    assert result.ok
    assert result.error == ""


@_needs_r
def test_invalid_r_fails_with_a_message():
    result = RAdapter().validate_syntax("if (TRUE) {\n")

    assert not result.ok
    assert result.error


# --- output_names (pure Python, no R needed) ---

def test_output_names_covers_the_common_assignment_forms():
    code = "\n".join(
        [
            "x <- 1",
            "y = 2",
            "z <<- 3",
            "4 -> w",
            "5 ->> v",
            'assign("u", 6)',
        ]
    )

    assert RAdapter().output_names(code) == ["x", "y", "z", "w", "v", "u"]


def test_output_names_ignores_function_argument_equals():
    # `mean(x = data)` binds nothing at top level; only `res <-` does.
    assert RAdapter().output_names("res <- mean(x = data)") == ["res"]


def test_output_names_skips_commented_and_dedupes():
    code = "a <- 1  # a comment with x <- 2\na <- 3"

    assert RAdapter().output_names(code) == ["a"]


# --- render_replay (Design B) ---

def _request(tmp_path, target_code="x <- 1", output_names=None, prefix_cells=None):
    return ReplayRequest(
        prefix_cells=prefix_cells if prefix_cells is not None else [],
        target_code=target_code,
        interval=0.05,
        output_names=output_names if output_names is not None else [],
        session_path=str(tmp_path / "s.zip"),
        fingerprint_path=str(tmp_path / "f.json"),
        output_path=str(tmp_path / "o"),
        work_dir=str(tmp_path),
    )


def test_render_replay_raises_without_rscript(tmp_path):
    with pytest.raises(CapabilityNotSupported):
        RAdapter(rscript="").render_replay(_request(tmp_path))


def test_render_replay_builds_a_harness_command(tmp_path):
    artifact = RAdapter(rscript="/usr/bin/Rscript").render_replay(_request(tmp_path))

    # The command drives the shared harness, which launches Rscript on the
    # generated script; perfmonitor and the export live in the harness, not here.
    assert artifact.script_path.endswith(".R")
    assert "jumper_extension.adapters.ai_reviewer.benchmark.harness" in artifact.command
    assert "--run" in artifact.command
    run_index = artifact.command.index("--run")
    assert "/usr/bin/Rscript" in artifact.command[run_index + 1]
    assert artifact.script_path in artifact.command[run_index + 1]


@_needs_r
def test_end_to_end_r_replay_times_and_fingerprints(tmp_path):
    from jumper_extension.adapters.ai_reviewer.benchmark.runner import BenchmarkRunner

    runner = BenchmarkRunner(
        prefix_cells=[{"index": 0, "raw_cell": "base <- 5", "cell_magics": []}],
        interval=0.05,
        adapter=RAdapter(),
        work_dir=str(tmp_path),
    )
    outcome = runner.run_once(
        "v <- c(1.0, 2.0, 3.0, 4.0)\ns <- base + sum(v)\nSys.sleep(0.15)",
        tag="run0",
        timeout=30,
    )

    assert outcome.ok
    assert outcome.duration_s > 0
    # Fingerprints are captured on the R side in the shared schema.
    assert outcome.fingerprints["s"]["kind"] == "scalar"
    assert outcome.fingerprints["v"]["kind"] == "array"
    assert outcome.fingerprints["v"]["shape"] == [4]


@_needs_r
def test_end_to_end_r_replay_reports_a_failing_cell(tmp_path):
    from jumper_extension.adapters.ai_reviewer.benchmark.runner import BenchmarkRunner

    runner = BenchmarkRunner(
        prefix_cells=[],
        interval=0.05,
        adapter=RAdapter(),
        work_dir=str(tmp_path),
    )
    outcome = runner.run_once("stop('boom')", tag="bad", timeout=30)

    assert not outcome.ok
    assert outcome.error
