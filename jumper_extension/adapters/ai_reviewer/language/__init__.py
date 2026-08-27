"""Per-language adapters for the AI-reviewer benchmark.

``base`` defines the ``LanguageAdapter`` interface and its capability names;
``registry`` resolves a cell's recorded language to an adapter, with a
no-capability fallback for anything unregistered.
"""
from jumper_extension.adapters.ai_reviewer.language.base import (
    RUN,
    VALIDATE_SYNTAX,
    CapabilityNotSupported,
    LanguageAdapter,
    ReplayArtifact,
    ReplayRequest,
    SyntaxResult,
)
from jumper_extension.adapters.ai_reviewer.language.registry import (
    DEFAULT_LANGUAGE,
    FallbackAdapter,
    get_adapter,
    register_adapter,
    resolve_language,
)

# Register the built-in adapters on first import of the package, so a bare
# `get_adapter("python")` works without callers wiring anything up.
from jumper_extension.adapters.ai_reviewer.language.python_adapter import PythonAdapter
from jumper_extension.adapters.ai_reviewer.language.r_adapter import RAdapter

register_adapter(PythonAdapter())
register_adapter(RAdapter())

__all__ = [
    "RUN",
    "VALIDATE_SYNTAX",
    "CapabilityNotSupported",
    "LanguageAdapter",
    "ReplayArtifact",
    "ReplayRequest",
    "SyntaxResult",
    "DEFAULT_LANGUAGE",
    "FallbackAdapter",
    "PythonAdapter",
    "RAdapter",
    "get_adapter",
    "register_adapter",
    "resolve_language",
]
