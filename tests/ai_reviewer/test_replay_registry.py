import logging

import pytest

from jumper_extension.adapters.ai_reviewer.benchmark.replay import (
    FULL,
    FullReplayStrategy,
    PrepareOutcome,
    ReplayContext,
    ReplayResult,
    ReplayStrategy,
    available_modes,
    register_strategy,
    resolve_strategy,
)
from jumper_extension.adapters.ai_reviewer.benchmark.replay import registry
from jumper_extension.adapters.ai_reviewer.benchmark.runner import BenchmarkRunner
from jumper_extension.adapters.ai_reviewer.language import get_adapter


@pytest.fixture
def context(tmp_path) -> ReplayContext:
    return ReplayContext(
        prefix_cells=[{"index": 0, "raw_cell": "x = 1", "cell_magics": []}],
        interval=0.05,
        work_dir=str(tmp_path),
        adapter=get_adapter("python"),
    )


@pytest.fixture
def clean_registry():
    """Let a test register a strategy without leaking it into the next one."""
    previous = dict(registry._STRATEGIES)
    yield
    registry._STRATEGIES.clear()
    registry._STRATEGIES.update(previous)


class _PythonOnly(ReplayStrategy):
    """Stand-in for fork/dill: serves Python, and nothing else."""
    name = "python_only"
    languages = frozenset({"python"})

    def replay(self, code: str, tag: str, timeout: float | None) -> ReplayResult:
        raise AssertionError("these tests never reach a replay")


class _CannotStart(ReplayStrategy):
    """Stand-in for a zygote that will not boot, or a checkpoint that will not dump."""
    name = "cannot_start"

    def prepare(self) -> PrepareOutcome:
        return PrepareOutcome(False, "no zygote today")

    def replay(self, code: str, tag: str, timeout: float | None) -> ReplayResult:
        raise AssertionError("a strategy that failed to prepare must not be replayed")


def test_full_is_always_available(context):
    assert FULL in available_modes()
    assert isinstance(resolve_strategy(FULL, context), FullReplayStrategy)


def test_no_mode_means_full(context):
    assert isinstance(resolve_strategy(None, context), FullReplayStrategy)


def test_unbuilt_mode_falls_back_to_full_with_a_reason(context, caplog):
    with caplog.at_level(logging.WARNING, logger="extension"):
        strategy = resolve_strategy("fork", context)

    assert isinstance(strategy, FullReplayStrategy)
    assert any("not available in this install" in r.message for r in caplog.records)


def test_registered_mode_is_used_for_a_language_it_serves(context, clean_registry):
    register_strategy(_PythonOnly)

    assert isinstance(resolve_strategy("python_only", context), _PythonOnly)


def test_registered_mode_falls_back_for_a_language_it_cannot_serve(
    context,
    clean_registry,
    caplog,
):
    register_strategy(_PythonOnly)
    context.adapter = get_adapter("r")

    with caplog.at_level(logging.WARNING, logger="extension"):
        strategy = resolve_strategy("python_only", context)

    assert isinstance(strategy, FullReplayStrategy)
    assert any("only serves" in r.message for r in caplog.records)


def test_target_lands_after_the_prefix_by_default(context):
    strategy = resolve_strategy(FULL, context)

    assert strategy.target_cell_index == len(context.prefix_cells)


def test_a_strategy_with_no_setup_prepares_and_closes_idempotently(context):
    strategy = resolve_strategy(FULL, context)

    assert strategy.prepare() == PrepareOutcome(True)
    strategy.close()
    strategy.close()


def test_runner_falls_back_to_full_when_a_strategy_cannot_prepare(
    tmp_path,
    clean_registry,
    caplog,
):
    register_strategy(_CannotStart)
    runner = BenchmarkRunner(
        prefix_cells=[],
        interval=0.05,
        work_dir=str(tmp_path),
        replay_mode="cannot_start",
    )
    assert isinstance(runner.strategy, _CannotStart)

    with caplog.at_level(logging.WARNING, logger="extension"):
        runner._ensure_prepared()

    assert isinstance(runner.strategy, FullReplayStrategy)
    assert any("no zygote today" in r.message for r in caplog.records)


def test_runner_prepares_only_once(tmp_path, clean_registry):
    calls = []

    class _Counting(ReplayStrategy):
        name = "counting"

        def prepare(self) -> PrepareOutcome:
            calls.append(1)
            return PrepareOutcome(True)

        def replay(self, code: str, tag: str, timeout: float | None) -> ReplayResult:
            raise AssertionError("not reached")

    register_strategy(_Counting)
    runner = BenchmarkRunner(
        prefix_cells=[],
        interval=0.05,
        work_dir=str(tmp_path),
        replay_mode="counting",
    )

    runner._ensure_prepared()
    runner._ensure_prepared()

    assert calls == [1]
