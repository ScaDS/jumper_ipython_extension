"""Replay strategies for the AI-reviewer benchmark.

``base`` defines the ``ReplayStrategy`` interface and the mode names;
``full`` is the always-correct replay every other mode falls back to;
``registry`` resolves a configured mode to a strategy, degrading with a reason
when the mode is unbuilt or cannot serve the cell's language.
"""
from jumper_extension.adapters.ai_reviewer.benchmark.replay.base import (
    DILL,
    FORK,
    FULL,
    PrepareOutcome,
    ReplayContext,
    ReplayResult,
    ReplayStrategy,
    tail,
)
from jumper_extension.adapters.ai_reviewer.benchmark.replay.full import FullReplayStrategy
from jumper_extension.adapters.ai_reviewer.benchmark.replay.registry import (
    available_modes,
    register_strategy,
    resolve_strategy,
)

__all__ = [
    "DILL",
    "FORK",
    "FULL",
    "FullReplayStrategy",
    "PrepareOutcome",
    "ReplayContext",
    "ReplayResult",
    "ReplayStrategy",
    "available_modes",
    "register_strategy",
    "resolve_strategy",
    "tail",
]
