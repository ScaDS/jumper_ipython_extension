"""Look up the adapter for a cell's language, with a safe fallback.

Adapters register themselves by language name; lookup is case-insensitive and
defaults to Python, which is also what legacy history rows - written before the
``language`` field existed - carry. An unrecognised language yields a fallback
that claims no capabilities, so the benchmark skips its checks with a reason
instead of guessing how to run it.
"""
from jumper_extension.adapters.ai_reviewer.language.base import (
    CapabilityNotSupported,
    LanguageAdapter,
    ReplayArtifact,
    ReplayRequest,
    SyntaxResult,
)

DEFAULT_LANGUAGE = "python"

_REGISTRY: dict[str, LanguageAdapter] = {}


def register_adapter(adapter: LanguageAdapter) -> None:
    """Register *adapter* under its (lower-cased) language name."""
    _REGISTRY[adapter.language.lower()] = adapter


def get_adapter(language: str | None) -> LanguageAdapter:
    """The adapter for *language*, or a no-capability fallback for the unknown."""
    key = (language or DEFAULT_LANGUAGE).lower()
    adapter = _REGISTRY.get(key)
    if adapter is not None:
        return adapter
    return FallbackAdapter(key)


def resolve_language(language: str | None) -> str:
    """Normalise a cell's recorded language; Python when absent (legacy rows)."""
    return str(language).lower() if language else DEFAULT_LANGUAGE


class FallbackAdapter(LanguageAdapter):
    """Stand-in for a language with no registered adapter: does nothing, safely.

    It claims no capabilities, so a well-behaved caller never reaches these
    methods; they raise rather than fabricate a result if one slips through.
    """
    caps = frozenset()

    def __init__(self, language: str):
        self.language = language

    def validate_syntax(self, code: str) -> SyntaxResult:
        raise CapabilityNotSupported(
            f"no adapter for language {self.language!r}: cannot validate syntax"
        )

    def output_names(self, code: str) -> list[str]:
        raise CapabilityNotSupported(
            f"no adapter for language {self.language!r}: cannot extract output names"
        )

    def render_replay(self, request: ReplayRequest) -> ReplayArtifact:
        raise CapabilityNotSupported(
            f"no adapter for language {self.language!r}: cannot render a replay"
        )
