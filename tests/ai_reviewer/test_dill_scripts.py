"""The scripts the dill replay generates, checked without needing dill itself.

Rendering is pure text work, so these run in every install - a build that cannot
even write a valid checkpoint script should not go green just because the
optional dependency is absent.
"""
from jumper_extension.adapters.ai_reviewer.benchmark.script import (
    build_checkpoint_script,
    build_restore_script,
)

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


def test_checkpoint_script_runs_the_prefix_in_its_own_namespace(tmp_path):
    source = open(build_checkpoint_script(_PREFIX, _PATHS, 1024, str(tmp_path / "cp.py"))).read()

    assert "_jumper.new_state()" in source          # not __main__: the user's own module
    assert "_jumper.run_cell(_jumper_state," in source   # same hooks as the full replay
    assert "_jumper.checkpoint(" in source
    # It measures nothing and exports nothing: this process only builds state.
    assert "perfmonitor_start" not in source
    assert "export_session" not in source


def test_checkpoint_script_passes_prefix_magics_along(tmp_path):
    prefix = [{
        "index": 0,
        "raw_cell": "%perfmonitor_start 0.05",
        "cell_magics": ["%perfmonitor_start 0.05"],
    }]
    source = open(build_checkpoint_script(prefix, _PATHS, 1024, str(tmp_path / "cp.py"))).read()

    # The adapter a captured magic will call is attached before any cell runs.
    assert "_jumper.attach_adapter(_jumper_state" in source
    assert "'%perfmonitor_start 0.05'" in source


def test_restore_script_loads_before_it_measures_and_seeds_last(tmp_path):
    source = open(
        build_restore_script(
            _PATHS,
            target_code="total = data.sum()",
            interval=0.05,
            fingerprint_names=["total"],
            session_path="/tmp/s.zip",
            fingerprint_path="/tmp/f.json",
            restore_report_path="/tmp/r.json",
            output_path=str(tmp_path / "restore.py"),
        )
    ).read()

    order = [
        source.index("_jumper.restore(_jumper_paths)"),
        source.index("_jumper.attach_adapter(_jumper_state"),
        source.index("perfmonitor_start"),
        source.index("_jumper.restore_rng"),
        source.index('"cell_started"'),
        source.index("_jumper.run_cell(_jumper_state"),
        source.index('"cell_finished"'),
        source.index("export_session"),
        source.index('"completed"'),
    ]
    assert order == sorted(order)
    # One cell runs here, the one under test: the prefix is restored, not replayed.
    assert source.count("_jumper.run_cell(") == 1
