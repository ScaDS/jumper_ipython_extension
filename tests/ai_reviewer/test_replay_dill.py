import os
import threading

import pytest

dill = pytest.importorskip("dill", reason="the dill replay mode needs the optional dill dependency")

# Imported after the skip on purpose: these pull in dill, so an install with the
# test extra but without [ai] must not fail collection.
from jumper_extension.adapters.ai_reviewer.benchmark import dill_state  # noqa: E402
from jumper_extension.adapters.ai_reviewer.benchmark.replay import (  # noqa: E402
    ReplayContext,
    StrategyChanged,
)
from jumper_extension.adapters.ai_reviewer.benchmark.replay.dill import (  # noqa: E402
    DillReplayStrategy,
)
from jumper_extension.adapters.ai_reviewer.benchmark.replay.full import (  # noqa: E402
    FullReplayStrategy,
)
from jumper_extension.adapters.ai_reviewer.benchmark.runner import BenchmarkRunner  # noqa: E402
from jumper_extension.adapters.ai_reviewer.language import get_adapter  # noqa: E402

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

        meta = dill_state.read_meta(runner.strategy._paths["meta"])
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
    state_dir = runner.strategy._state_dir
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

    monkeypatch.setattr(dill_mode, "_max_checkpoint_bytes", lambda work_dir: 1024)
    runner = build_runner(tmp_path, _PREFIX)
    try:
        outcome = runner.run_once("total = float(data.sum())", tag="v1")

        assert isinstance(runner.strategy, FullReplayStrategy)
        assert outcome.ok
    finally:
        runner.close()


def test_a_checkpoint_that_will_not_load_is_not_blamed_on_the_cell(tmp_path):
    runner = build_runner(tmp_path, _PREFIX)
    try:
        runner._ensure_prepared()
        checkpoint = runner.strategy._paths["checkpoint"]
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
        strategy = runner.strategy
        dill_state.write_phase(os.path.join(strategy._state_dir, "v1.phase"), "completed")
        with open(strategy._paths["checkpoint"], "wb") as handle:
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


def test_a_handle_hidden_on_an_object_is_refused(tmp_path):
    """The shallow screen cannot see inside an object; the pickler guard can.

    Restoring such a handle reopens the file in its original mode - `w+` means
    truncation - so the data the prefix wrote through it would be destroyed on
    every measurement, and the benchmark would report success while doing it.
    """
    victim = tmp_path / "user_data.txt"
    prefix = [{
        "index": 0,
        "raw_cell": (
            "class Holder:\n"
            "    pass\n"
            "holder = Holder()\n"
            f"holder.handle = open({str(victim)!r}, 'w+')\n"
            "holder.handle.write('PRECIOUS USER DATA')\n"
            "holder.handle.flush()"
        ),
        "cell_magics": [],
    }]
    runner = build_runner(tmp_path, prefix)
    try:
        outcome = runner.run_once("x = 1", tag="v1")

        assert isinstance(runner.strategy, FullReplayStrategy)
        assert outcome.ok
        # Written by the prefix, and still there: nothing reopened it.
        assert victim.read_text() == "PRECIOUS USER DATA"
    finally:
        runner.close()


def test_a_callers_own_files_are_left_alone(tmp_path):
    """The strategy makes its own directory and removes exactly that one."""
    mine = tmp_path / "dill_state"
    mine.mkdir()
    (mine / "user-owned.txt").write_text("mine")

    runner = build_runner(tmp_path, _PREFIX)
    runner.run_once("total = float(data.sum())", tag="v1")
    runner.close()

    assert (mine / "user-owned.txt").read_text() == "mine"


def test_a_prefix_full_of_magics_still_checkpoints(tmp_path):
    """`%perfmonitor_start` is in nearly every prefix, and it renders to an
    adapter call - so the checkpoint process has to bring an adapter."""
    prefix = [
        {"index": 0, "raw_cell": "%perfmonitor_start 0.05", "cell_magics": ["%perfmonitor_start 0.05"]},
        {"index": 1, "raw_cell": "import numpy as np\nvalues = np.arange(1000)", "cell_magics": []},
    ]
    runner = build_runner(tmp_path, prefix)
    try:
        outcome = runner.run_once("total = int(values.sum())", tag="v1")

        assert isinstance(runner.strategy, DillReplayStrategy)
        assert outcome.ok, outcome.error
        assert outcome.fingerprints["total"]["value"] == 499500
    finally:
        runner.close()


def test_a_prefix_that_shadows_the_helpers_keeps_its_own_values(tmp_path):
    """Bookkeeping names are removed by identity, so a user's `_jumper` survives."""
    prefix = [{
        "index": 0,
        "raw_cell": "_jumper = 'mine'\ndump = 42\nmagic = 'kept'",
        "cell_magics": [],
    }]
    runner = build_runner(tmp_path, prefix)
    try:
        outcome = runner.run_once("kept = _jumper + str(dump) + magic", tag="v1")

        assert isinstance(runner.strategy, DillReplayStrategy)
        assert outcome.ok, outcome.error
    finally:
        runner.close()


def test_a_missing_rng_artifact_is_not_silently_ignored(tmp_path):
    """Losing it means every measurement reseeds from entropy, and a correct
    variant gets reported as computing something else."""
    prefix = [{"index": 0, "raw_cell": "import random\nrandom.seed(99)", "cell_magics": []}]
    runner = build_runner(tmp_path, prefix)
    try:
        runner._ensure_prepared()
        os.remove(runner.strategy._paths["rng"])

        with pytest.raises(StrategyChanged):
            runner.run_once("draw = random.random()", tag="v1")
    finally:
        runner.close()


def test_a_failure_after_the_cell_is_not_the_cells_fault(tmp_path, context):
    """Fingerprinting, stopping the monitor and exporting all happen after the
    cell has run; a failure there belongs to the benchmark, not the suggestion."""
    from jumper_extension.adapters.ai_reviewer.benchmark.models import FAILED
    from jumper_extension.adapters.ai_reviewer.benchmark.replay.base import ReplayResult

    strategy = DillReplayStrategy(context)
    phase_path = str(tmp_path / "v1.phase")
    failed = ReplayResult(status=FAILED, error="No space left on device")

    for phase, expected in [
        ("cell_started", False),
        ("cell_finished", True),
        ("exporting", True),
        ("loading", True),
        ("setup", True),
    ]:
        dill_state.write_phase(phase_path, phase)
        blamed = strategy._blame(failed, phase_path)
        assert blamed.strategy_broken is expected, phase


def test_an_unreadable_prepare_log_still_falls_back(tmp_path):
    """A prefix that prints one invalid byte used to crash preparation itself."""
    from jumper_extension.adapters.ai_reviewer.benchmark.replay import dill as dill_mode

    log = tmp_path / "checkpoint.log"
    log.write_bytes(b"traceback with a bad byte: \xff\xfe and more")

    assert "bad byte" in dill_mode._read_tail(str(log))
