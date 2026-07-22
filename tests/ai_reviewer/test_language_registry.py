import pytest

from jumper_extension.adapters.ai_reviewer.language import (
    RUN,
    VALIDATE_SYNTAX,
    VERIFY_RESULTS,
    CapabilityNotSupported,
    FallbackAdapter,
    LanguageAdapter,
    ReplayArtifact,
    ReplayRequest,
    SyntaxResult,
    get_adapter,
    register_adapter,
    resolve_language,
)
from jumper_extension.adapters.ai_reviewer.language import registry


class _StubAdapter(LanguageAdapter):
    language = "stub"
    caps = frozenset({VALIDATE_SYNTAX, VERIFY_RESULTS})

    def validate_syntax(self, code):
        return SyntaxResult(ok=True)

    def output_names(self, code):
        return ["x"]

    def render_replay(self, request):
        return ReplayArtifact(script_path="s", command=["run", "s"])


@pytest.fixture(autouse=True)
def _clean_registry():
    """Each test sees an empty registry, restored afterwards."""
    saved = dict(registry._REGISTRY)
    registry._REGISTRY.clear()
    try:
        yield
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(saved)


# --- resolve_language ---

def test_resolve_language_normalises_case():
    assert resolve_language("R") == "r"


def test_resolve_language_defaults_to_python_when_absent():
    assert resolve_language(None) == "python"
    assert resolve_language("") == "python"


# --- get_adapter ---

def test_get_adapter_returns_registered_adapter_case_insensitively():
    adapter = _StubAdapter()
    register_adapter(adapter)

    assert get_adapter("stub") is adapter
    assert get_adapter("STUB") is adapter


def test_get_adapter_falls_back_for_unknown_language():
    adapter = get_adapter("cobol")

    assert isinstance(adapter, FallbackAdapter)
    assert adapter.language == "cobol"
    assert adapter.caps == frozenset()


def test_get_adapter_defaults_to_python_key_when_language_is_none():
    adapter = get_adapter(None)

    assert isinstance(adapter, FallbackAdapter)
    assert adapter.language == "python"


# --- capability model ---

def test_supports_reflects_declared_caps():
    adapter = _StubAdapter()

    assert adapter.supports(VALIDATE_SYNTAX)
    assert adapter.supports(VERIFY_RESULTS)
    assert not adapter.supports(RUN)


# --- fallback safety ---

def test_fallback_seams_raise_rather_than_fabricate():
    adapter = FallbackAdapter("cobol")
    request = ReplayRequest(
        prefix_cells=[],
        target_code="x",
        interval=0.05,
        output_names=[],
        session_path="a",
        fingerprint_path="b",
        output_path="c",
        work_dir="d",
    )

    for call in (
        lambda: adapter.validate_syntax("x"),
        lambda: adapter.output_names("x"),
        lambda: adapter.render_replay(request),
    ):
        with pytest.raises(CapabilityNotSupported):
            call()
