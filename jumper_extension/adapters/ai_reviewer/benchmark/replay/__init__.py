"""Replay strategies for the AI-reviewer benchmark.

``base`` defines the ``ReplayStrategy`` interface and the mode names;
``full`` is the always-correct replay every other mode falls back to;
``fork`` rebuilds the prefix state once and forks a child per measurement;
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

# Register the built-in strategies on first import of the package, so a bare
# `resolve_strategy("fork", ...)` works without callers wiring anything up.
from jumper_extension.adapters.ai_reviewer.benchmark.replay.fork import ForkReplayStrategy

register_strategy(ForkReplayStrategy)

__all__ = [
    "DILL",
    "FORK",
    "FULL",
    "ForkReplayStrategy",
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
