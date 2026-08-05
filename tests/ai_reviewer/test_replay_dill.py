import json
import os
import threading

import pytest

from jumper_extension.adapters.ai_reviewer.benchmark import dill_state
from jumper_extension.adapters.ai_reviewer.benchmark.replay import ReplayContext, StrategyChanged
from jumper_extension.adapters.ai_reviewer.benchmark.replay.dill import DillReplayStrategy
from jumper_extension.adapters.ai_reviewer.benchmark.replay.full import FullReplayStrategy
from jumper_extension.adapters.ai_reviewer.benchmark.runner import BenchmarkRunner
from jumper_extension.adapters.ai_reviewer.language import get_adapter

dill = pytest.importorskip("dill", reason="the dill replay mode needs the optional dill dependency")

_PREFIX = [
    {"index": 0, "raw_cell": "import numpy as np", "cell_magics": []},
    {"index": 1, "raw_cell": "np.random.seed(7)\ndata = np.random.random(50_000)", "cell_magics": []},
]

_PATHS = {
    "checkpoint": "/tmp/state/checkpoint.pkl",
    "meta": "/tmp/state/meta.json",
    "rng": "/tmp/state/rng.pkl",
    "phase": "/tmp/state/tag.phase",
}


@pytest.fixture
def context(tmp_path) -> ReplayContext:
    return ReplayContext(
        prefix_cells=_PREFIX,
        interval=0.05,
        work_dir=str(tmp_path),
        adapter=get_adapter("python"),
    )


def build_runner(tmp_path, prefix: list) -> BenchmarkRunner:
    return BenchmarkRunner(
        prefix_cells=prefix,
        interval=0.05,
        work_dir=str(tmp_path),
        replay_mode="dill",
    )


# --- the state module ------------------------------------------------------

def test_unsafe_bindings_names_what_a_checkpoint_would_ruin(tmp_path):
    handle = open(tmp_path / "notes.txt", "w+")
    try:
        namespace = {
            "data": [1, 2, 3],
            "handle": handle,
            "guard": threading.Lock(),
            "nested": {"inner": [threading.Lock()]},
            "__builtins__": {},
        }
        found = dill_state.unsafe_bindings(namespace)
    finally:
        handle.close()

    reported = " ".join(found)
    assert "handle" in reported and "guard" in reported
    assert "nested['inner'][0]" in reported  # containers are walked, not trusted
    assert "data" not in reported


def test_in_memory_buffers_are_not_refused():
    import io

    assert dill_state.unsafe_bindings({"buffer": io.BytesIO(b"x")}) == []


def test_rng_state_survives_a_capture_and_restore(tmp_path):
    import random

    random.seed(11)
    expected = [random.random() for _ in range(3)]

    random.seed(11)
    path = str(tmp_path / "rng.pkl")
    with open(path, "wb") as handle:
        dill.dump(dill_state.capture_rng(), handle)
    random.random()  # move the stream on, as a restore's setup would

    restored = dill_state.restore_rng(path)
    assert "random" in restored
    assert [random.random() for _ in range(3)] == expected


def test_environment_mismatch_reports_the_difference():
    assert dill_state.environment_mismatch({"environment": dill_state.environment()}) == ""
    stale = {"environment": {**dill_state.environment(), "dill": "0.0.1"}}
    assert "dill" in dill_state.environment_mismatch(stale)


def test_read_meta_survives_a_missing_or_broken_file(tmp_path):
    assert dill_state.read_meta(str(tmp_path / "absent.json")) == {}
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert dill_state.read_meta(str(broken)) == {}


# --- the strategy end to end ----------------------------------------------

def test_the_target_is_the_only_cell_in_the_history(context):
    assert DillReplayStrategy(context).target_cell_index == 0


def test_dill_only_serves_python(context):
    strategy = DillReplayStrategy(context)
    assert strategy.supports(get_adapter("python"))
    assert not strategy.supports(get_adapter("r"))


def test_a_measurement_restores_the_prefix_state(tmp_path):
    runner = build_runner(tmp_path, _PREFIX)
    try:
        outcome = runner.run_once("total = float(data.sum())", tag="v1")

        assert isinstance(runner.strategy, DillReplayStrategy)  # no fallback happened
        assert outcome.ok, outcome.error
        assert outcome.duration_s is not None
        assert outcome.fingerprints["total"]["kind"] == "scalar"

        meta = dill_state.read_meta(os.path.join(str(tmp_path), "dill_state", "meta.json"))
        assert meta["prefix_s"] >= 0
        assert meta["environment"]["dill"] == dill.__version__
    finally:
        runner.close()


def test_two_measurements_draw_the_same_random_numbers(tmp_path):
    """The failure this guards: a restored process reseeds from OS entropy, every
    measurement computes different numbers, and a correct variant is reported as
    DIFFERS and sent to the repair loop."""
    runner = build_runner(tmp_path, _PREFIX)
    try:
        code = "draw = float(np.random.random())"
        first = runner.run_once(code, tag="v1")
        second = runner.run_once(code, tag="v2")

        assert first.ok and second.ok
        assert first.fingerprints["draw"] == second.fingerprints["draw"]
    finally:
        runner.close()


def test_the_checkpoint_directory_is_removed_on_close(tmp_path):
    runner = build_runner(tmp_path, _PREFIX)
    runner.run_once("total = float(data.sum())", tag="v1")
    state_dir = os.path.join(str(tmp_path), "dill_state")
    assert os.path.exists(state_dir)

    runner.close()
    runner.close()  # the contract requires this to be safe
    assert not os.path.exists(state_dir)


@pytest.mark.parametrize(
    "raw_cell, expected",
    [
        ("handle = open('scratch.txt', 'w+')", "an open file"),
        ("import threading\nguard = threading.Lock()", "_thread.lock"),
    ],
)
def test_state_that_would_be_silently_ruined_refuses_the_mode(tmp_path, raw_cell, expected):
    """Both of these pickle without complaining, and the file is even truncated
    on restore - so only a screen before the dump can catch them."""
    runner = build_runner(tmp_path, [{"index": 0, "raw_cell": raw_cell, "cell_magics": []}])
    try:
        outcome = runner.run_once("x = 1", tag="v1")

        assert isinstance(runner.strategy, FullReplayStrategy)
        assert outcome.ok  # the benchmark carries on, it does not fail
    finally:
        runner.close()


def test_a_value_that_will_not_pickle_falls_back(tmp_path):
    """A generator is not on the screen's list - it refuses honestly at the dump,
    and that failure has to end in the full replay rather than in a traceback."""
    prefix = [{"index": 0, "raw_cell": "stream = (i for i in range(3))", "cell_magics": []}]
    runner = build_runner(tmp_path, prefix)
    try:
        outcome = runner.run_once("x = 1", tag="v1")

        assert isinstance(runner.strategy, FullReplayStrategy)
        assert outcome.ok
    finally:
        runner.close()


def test_a_checkpoint_over_the_limit_falls_back(tmp_path, monkeypatch):
    from jumper_extension.adapters.ai_reviewer.benchmark.replay import dill as dill_mode

    monkeypatch.setattr(dill_mode, "_max_checkpoint_bytes", lambda: 1024)
    runner = build_runner(tmp_path, _PREFIX)
    try:
        outcome = runner.run_once("total = float(data.sum())", tag="v1")

        assert isinstance(runner.strategy, FullReplayStrategy)
        assert outcome.ok
        assert not os.path.exists(os.path.join(str(tmp_path), "dill_state", "checkpoint.pkl"))
    finally:
        runner.close()


def test_a_checkpoint_that_will_not_load_is_not_blamed_on_the_cell(tmp_path):
    runner = build_runner(tmp_path, _PREFIX)
    try:
        runner._ensure_prepared()
        checkpoint = os.path.join(str(tmp_path), "dill_state", "checkpoint.pkl")
        with open(checkpoint, "wb") as handle:
            handle.write(b"not a checkpoint")

        with pytest.raises(StrategyChanged):
            runner.run_once("x = 1", tag="v1")
        assert isinstance(runner.strategy, FullReplayStrategy)
    finally:
        runner.close()


def test_a_stale_phase_file_cannot_vouch_for_a_later_failure(tmp_path):
    """Tags repeat across repair attempts, so a `completed` left by the previous
    attempt would make a checkpoint failure look like the suggestion's fault."""
    runner = build_runner(tmp_path, _PREFIX)
    try:
        runner._ensure_prepared()
        state_dir = os.path.join(str(tmp_path), "dill_state")
        dill_state.write_phase(os.path.join(state_dir, "v1.phase"), "completed")
        with open(os.path.join(state_dir, "checkpoint.pkl"), "wb") as handle:
            handle.write(b"not a checkpoint")

        with pytest.raises(StrategyChanged):
            runner.run_once("x = 1", tag="v1")
    finally:
        runner.close()


def test_a_failing_cell_is_blamed_on_the_cell(tmp_path):
    runner = build_runner(tmp_path, _PREFIX)
    try:
        outcome = runner.run_once("raise ValueError('the suggestion is wrong')", tag="v1")

        assert not outcome.ok
        assert "ValueError" in outcome.error
        assert isinstance(runner.strategy, DillReplayStrategy)  # the mode is fine
    finally:
        runner.close()


# --- the cross-mode check --------------------------------------------------

def test_cross_check_is_skipped_when_the_full_replay_is_already_active(tmp_path):
    runner = BenchmarkRunner(prefix_cells=_PREFIX, interval=0.05, work_dir=str(tmp_path))

    assert runner.cross_check_baseline("total = float(data.sum())") is None


def test_cross_check_measures_the_baseline_the_other_way(tmp_path):
    runner = build_runner(tmp_path, _PREFIX)
    try:
        outcome = runner.cross_check_baseline("total = float(data.sum())")

        assert outcome is not None and outcome.ok
        assert outcome.fingerprints["total"]["kind"] == "scalar"
        # It ran on its own strategy and left the configured one untouched.
        assert isinstance(runner.strategy, DillReplayStrategy)
    finally:
        runner.close()
