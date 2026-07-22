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


def test_run_is_not_claimed_yet():
    # 5a exposes no timed run (and so no result verification); that is Phase 5b.
    from jumper_extension.adapters.ai_reviewer.language import RUN

    adapter = get_adapter("r")
    assert RUN not in adapter.caps


def test_capability_drops_when_rscript_is_missing():
    adapter = RAdapter(rscript="")

    assert adapter.caps == frozenset()
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


# --- render_replay is deferred to 5b ---

def test_render_replay_is_not_implemented_yet():
    request = ReplayRequest(
        prefix_cells=[],
        target_code="x <- 1",
        interval=0.05,
        output_names=[],
        session_path="s",
        fingerprint_path="f",
        output_path="o",
        work_dir="w",
    )
    with pytest.raises(CapabilityNotSupported):
        RAdapter().render_replay(request)
