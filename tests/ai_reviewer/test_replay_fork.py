import os

import pytest

from jumper_extension.adapters.ai_reviewer.benchmark.replay import ReplayContext
from jumper_extension.adapters.ai_reviewer.benchmark.replay.fork import ForkReplayStrategy
from jumper_extension.adapters.ai_reviewer.benchmark.replay.zygote import (
    gpu_blocker,
    prefault,
    probe_fork,
)
from jumper_extension.adapters.ai_reviewer.benchmark.script import build_prefix_script
from jumper_extension.adapters.ai_reviewer.language import get_adapter

_PREFIX = [
    {"index": 0, "raw_cell": "import numpy as np", "cell_magics": []},
    {"index": 1, "raw_cell": "data = np.arange(100_000, dtype=float)", "cell_magics": []},
]

posix_only = pytest.mark.skipif(os.name != "posix", reason="fork needs a POSIX platform")


@pytest.fixture
def context(tmp_path) -> ReplayContext:
    return ReplayContext(
        prefix_cells=_PREFIX,
        interval=0.05,
        work_dir=str(tmp_path),
        adapter=get_adapter("python"),
    )


def test_prefix_script_has_the_cells_but_no_header_or_footer(tmp_path):
    path = build_prefix_script(_PREFIX, str(tmp_path / "prefix.py"))
    source = open(path).read()

    assert "import numpy as np" in source
    assert "magic_adapter.on_pre_run_cell" in source  # driven through the same hooks
    # The zygote brings its own adapter and owns the export, so neither belongs here.
    assert "build_perfmonitor_magic_adapter" not in source
    assert "export_session" not in source


def test_fork_only_serves_python(context):
    strategy = ForkReplayStrategy(context)

    assert strategy.supports(get_adapter("python"))
    assert not strategy.supports(get_adapter("r"))


def test_fork_refuses_off_posix(context, monkeypatch):
    monkeypatch.setattr(os, "name", "nt")

    outcome = ForkReplayStrategy(context).prepare()

    assert not outcome.ok
    assert "POSIX" in outcome.reason


def test_replay_before_prepare_is_the_strategy_failing_not_the_code(context):
    """A dead zygote must never be reported as a failing suggestion.

    Otherwise the repair loop is handed perfectly good code to fix, three times,
    for every variant.
    """
    result = ForkReplayStrategy(context).replay("x = 1", "t0", None)

    assert not result.ok
    assert result.strategy_broken


def test_no_gpu_module_imported_is_no_blocker():
    assert gpu_blocker() == ""


@posix_only
def test_prefault_walks_this_process_pages():
    pytest.importorskip("numpy")

    assert prefault() > 0


def test_prefault_survives_an_unreadable_maps_file(tmp_path):
    assert prefault(str(tmp_path / "absent")) == 0


@posix_only
def test_probe_answers_with_thread_counts_not_timings(tmp_path):
    """The verdict is structural on purpose.

    An earlier version decided by comparing timings across a fork and refused a
    healthy machine in 6 runs out of 20. Timings are still reported - they are
    the only way an unknown slowdown would surface - but they must not be what
    the answer turns on.
    """
    pytest.importorskip("numpy")

    probe = probe_fork(str(tmp_path))

    assert isinstance(probe["ok"], bool)
    if not probe["ok"]:
        # A bare refusal leaves no way to tell degradation from a noisy machine.
        assert probe["detail"]
        return
    assert "/" in probe["threads"], "reports what the child recovered of the parent's"
    assert set(probe["timings"]) == {"compute", "memory"}
    # The unit the per-measurement fault warning is stated in, measured here
    # rather than picked in advance.
    assert probe["fault_cost_s"] >= 0.0


@posix_only
def test_fork_replays_a_cell_end_to_end(context):
    """The whole path: boot a zygote, fork it, and read a session back."""
    pytest.importorskip("numpy")
    strategy = ForkReplayStrategy(context)
    outcome = strategy.prepare()
    if not outcome.ok:
        pytest.skip(f"this machine cannot be forked safely: {outcome.reason}")

    try:
        good = strategy.replay("total = float(data.sum())", "good", None)
        bad = strategy.replay("total = data.nope()", "bad", None)
        # The zygote has to survive a failed child and keep serving.
        again = strategy.replay("total = float(data.sum())", "again", None)
    finally:
        strategy.close()

    assert good.ok and os.path.exists(good.session_path)
    assert os.path.exists(good.fingerprint_path)
    assert not bad.ok and not bad.strategy_broken
    assert "nope" in bad.error
    assert again.ok


@posix_only
def test_a_runaway_cell_is_killed_and_the_zygote_survives(context):
    pytest.importorskip("numpy")
    strategy = ForkReplayStrategy(context)
    outcome = strategy.prepare()
    if not outcome.ok:
        pytest.skip(f"this machine cannot be forked safely: {outcome.reason}")

    try:
        killed = strategy.replay("import time; time.sleep(30)", "slow", 2.0)
        after = strategy.replay("total = float(data.sum())", "after", None)
    finally:
        strategy.close()

    assert killed.status == "timeout"
    assert not killed.strategy_broken
    assert after.ok, "killing a child must not take the zygote with it"
