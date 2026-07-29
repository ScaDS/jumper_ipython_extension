"""Pick the replay strategy for one benchmark, and say why it is not the asked-for one.

Two things gate a mode: whether it is built into this install at all, and
whether it can serve the cell's language. Either way the answer is never a
failure - the full replay is always correct, so an unavailable mode degrades to
it with a warning naming the reason. This mirrors how ``checks.resolve_checks``
treats a step the adapter cannot perform: skip honestly, never pretend.
"""
import logging

from jumper_extension.adapters.ai_reviewer.benchmark.replay.base import (
    FULL,
    ReplayContext,
    ReplayStrategy,
)
from jumper_extension.adapters.ai_reviewer.benchmark.replay.full import FullReplayStrategy

logger = logging.getLogger("extension")

# Modes beyond FULL register themselves here on import of their own module, so a
# mode that is configured but not built simply falls back instead of raising.
_STRATEGIES: dict[str, type[ReplayStrategy]] = {}


def register_strategy(strategy_type: type[ReplayStrategy]):
    """Register *strategy_type* under its ``name``."""
    _STRATEGIES[strategy_type.name.lower()] = strategy_type


def available_modes() -> list[str]:
    """Every mode that can currently be selected, full replay included."""
    return sorted({FULL, *_STRATEGIES})


def resolve_strategy(mode: str | None, context: ReplayContext) -> ReplayStrategy:
    """The strategy for *mode*, or the full replay when it cannot serve this cell."""
    requested = (mode or FULL).lower()
    if requested == FULL:
        return FullReplayStrategy(context)

    strategy_type = _STRATEGIES.get(requested)
    if strategy_type is None:
        return _fallback(requested, "not available in this install", context)

    strategy = strategy_type(context)
    if not strategy.supports(context.adapter):
        return _fallback(
            requested,
            f"it only serves {sorted(strategy.languages)}, "
            f"and this cell is {context.adapter.language!r}",
            context,
        )
    return strategy


def _fallback(requested: str, reason: str, context: ReplayContext) -> ReplayStrategy:
    logger.warning(
        f"[JUmPER]: benchmark replay mode {requested!r} unavailable ({reason}); "
        "falling back to the full replay."
    )
    return FullReplayStrategy(context)
