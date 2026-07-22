"""Decide which benchmark steps run for one cell, and say why the rest do not.

Two things gate each step: whether the user asked for it (config, overridden
per-run by ``--check``/``--skip-check``) and whether the cell's language adapter
can actually do it. A step the user turned off is skipped in silence - that was
the intent. A step the user left on but the adapter cannot perform is skipped
with a warning naming the reason, so a language that cannot, say, run yet still
gets its suggestions syntax-checked instead of failing outright.
"""
import logging
from dataclasses import dataclass

from jumper_extension.adapters.ai_reviewer.language import (
    RUN,
    VALIDATE_SYNTAX,
    VERIFY_RESULTS,
    LanguageAdapter,
)

logger = logging.getLogger("extension")

_NAMES = ("validate_syntax", "verify_results", "run")
_CAPABILITY = {
    "validate_syntax": VALIDATE_SYNTAX,
    "verify_results": VERIFY_RESULTS,
    "run": RUN,
}


@dataclass
class Decision:
    """Whether one step runs, and - when skipped despite being asked for - why."""
    active: bool
    reason: str = ""


@dataclass
class CheckPlan:
    validate_syntax: Decision
    verify_results: Decision
    run: Decision


def all_active() -> CheckPlan:
    """The default plan: every step on. Used when no plan is injected."""
    return CheckPlan(
        validate_syntax=Decision(True),
        verify_results=Decision(True),
        run=Decision(True),
    )


def resolve_checks(
    adapter: LanguageAdapter,
    config_checks,
    overrides: dict | None = None,
) -> CheckPlan:
    """Combine config, per-run overrides and adapter capability into a plan.

    *config_checks* is the ``AIBenchmarkChecksConfig`` (``.validate_syntax`` etc);
    *overrides* maps a step name to a forced on/off from the command line.
    """
    enabled = {name: getattr(config_checks, name) for name in _NAMES}
    enabled.update(overrides or {})

    decisions = {name: _decide(name, enabled[name], adapter) for name in _NAMES}

    # Verifying results means fingerprinting what an execution produced; with no
    # timed run there is nothing to fingerprint, so it cannot stand on its own.
    if decisions["verify_results"].active and not decisions["run"].active:
        decisions["verify_results"] = Decision(
            False, "requires the timed run, which is not running"
        )
    return CheckPlan(**decisions)


def _decide(name: str, enabled: bool, adapter: LanguageAdapter) -> Decision:
    if not enabled:
        return Decision(False)  # turned off on purpose - no warning
    if not adapter.supports(_CAPABILITY[name]):
        reason = f"no {name} capability for language {adapter.language!r}"
        logger.warning(f"[JUmPER]: benchmark {name} skipped: {reason}")
        return Decision(False, reason)
    return Decision(True)
