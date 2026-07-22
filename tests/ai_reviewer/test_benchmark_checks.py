import logging

from jumper_extension.adapters.ai_reviewer.benchmark.checks import (
    all_active,
    resolve_checks,
)
from jumper_extension.adapters.ai_reviewer.language import (
    RUN,
    VALIDATE_SYNTAX,
    LanguageAdapter,
    ReplayArtifact,
    SyntaxResult,
)
from jumper_extension.config.models import AIBenchmarkChecksConfig


class _Adapter(LanguageAdapter):
    def __init__(self, language, caps):
        self.language = language
        self.caps = frozenset(caps)

    def validate_syntax(self, code):
        return SyntaxResult(ok=True)

    def output_names(self, code):
        return []

    def render_replay(self, request):
        return ReplayArtifact(script_path="", command=[])


_ALL_CAPS = {VALIDATE_SYNTAX, RUN}


def _config(**overrides):
    return AIBenchmarkChecksConfig(**overrides)


def test_all_active_turns_every_step_on():
    plan = all_active()

    assert plan.validate_syntax.active
    assert plan.run.active


def test_config_enabled_and_adapter_capable_runs_every_step():
    plan = resolve_checks(_Adapter("python", _ALL_CAPS), _config(), overrides=None)

    assert plan.validate_syntax.active
    assert plan.run.active


def test_step_disabled_in_config_is_skipped_without_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="extension"):
        plan = resolve_checks(
            _Adapter("python", _ALL_CAPS),
            _config(validate_syntax=False),
            overrides=None,
        )

    assert not plan.validate_syntax.active
    assert plan.validate_syntax.reason == ""
    assert caplog.records == []


def test_enabled_step_without_capability_is_skipped_with_warning(caplog):
    # R that can validate syntax but cannot run yet: the run step degrades.
    adapter = _Adapter("r", {VALIDATE_SYNTAX})
    with caplog.at_level(logging.WARNING, logger="extension"):
        plan = resolve_checks(adapter, _config(), overrides=None)

    assert plan.validate_syntax.active
    assert not plan.run.active
    assert "no run capability for language 'r'" in plan.run.reason
    assert any("benchmark run skipped" in r.message for r in caplog.records)


def test_run_can_be_turned_off_leaving_only_the_syntax_gate():
    # The syntax gate stands on its own; disabling run is silent (deliberate).
    plan = resolve_checks(
        _Adapter("python", _ALL_CAPS),
        _config(run=False),
        overrides=None,
    )

    assert plan.validate_syntax.active
    assert not plan.run.active
    assert plan.run.reason == ""


def test_every_check_status_is_logged_at_debug_even_when_off(caplog):
    with caplog.at_level(logging.DEBUG, logger="extension"):
        resolve_checks(
            _Adapter("python", _ALL_CAPS),
            _config(validate_syntax=False),
            overrides=None,
        )

    debug = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("check validate_syntax for 'python': skipped (disabled)" in m for m in debug)
    assert any("check run for 'python': active" in m for m in debug)


def test_overrides_win_over_config():
    plan = resolve_checks(
        _Adapter("python", _ALL_CAPS),
        _config(),
        overrides={"run": False},
    )

    assert plan.validate_syntax.active
    assert not plan.run.active
