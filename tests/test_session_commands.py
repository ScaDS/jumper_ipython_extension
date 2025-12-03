import os
import json
import zipfile
from pathlib import Path

import pandas as pd

from jumper_extension.ipython.magics import PerfmonitorMagics
from jumper_extension.core.service import build_perfmonitor_magic_adapter


def test_export_session_directory(ipython, mock_cpu_only, temp_dir):
    magics = PerfmonitorMagics(ipython, build_perfmonitor_magic_adapter())
    # start to initialize monitor/visualizer/reporter
    magics.perfmonitor_start("")

    export_dir = Path(temp_dir) / "sess1"
    magics.export_session(str(export_dir))

    assert export_dir.exists(), "Export directory should be created"
    assert (export_dir / "manifest.json").exists(), "Manifest should be written"


def test_export_session_zip(ipython, mock_cpu_only, temp_dir):
    magics = PerfmonitorMagics(ipython, build_perfmonitor_magic_adapter())
    magics.perfmonitor_start("")

    zip_path = Path(temp_dir) / "sess2.zip"
    magics.export_session(str(zip_path))

    assert zip_path.exists(), "Zip archive should be created"
    with zipfile.ZipFile(zip_path, 'r') as zf:
        assert "manifest.json" in zf.namelist(), "Zip should contain manifest.json"


def _write_fake_session(dir_path: Path):
    dir_path.mkdir(parents=True, exist_ok=True)
    # minimal cell history
    ch = pd.DataFrame([
        {
            "cell_index": 0,
            "raw_cell": "x = 1",
            "start_time": 1.0,
            "end_time": 2.0,
            "duration": 1.0,
        }
    ])
    ch.to_csv(dir_path / "cell_history.csv", index=False)

    # minimal perf data
    perf = pd.DataFrame([
        {
            "time": 1.5,
            "memory": 1.0,
            "io_read_count": 0.0,
            "io_write_count": 0.0,
            "io_read": 0.0,
            "io_write": 0.0,
            "cpu_util_avg": 10.0,
            "cpu_util_min": 10.0,
            "cpu_util_max": 10.0,
        }
    ])
    perf.to_csv(dir_path / "perf_process.csv", index=False)

    manifest = {
        "version": "1.0",
        "app": {"name": "JUmPER", "version": "test"},
        "monitor": {
            "interval": 1.0,
            "start_time": 1.0,
            "stop_time": 2.0,
            "num_cpus": 1,
            "num_system_cpus": 1,
            "num_gpus": 0,
            "gpu_memory": 0.0,
            "gpu_name": "",
            "memory_limits": {"system": 8.0},
            "cpu_handles": [0],
            "pid": 123,
            "uid": 1000,
            "slurm_job": 0,
            "os": os.name,
            "python": "3.x",
        },
        "levels": ["process", "user", "system"],
        "schemas": {"perf": {"process": list(perf.columns)}, "cell_history": list(ch.columns)},
        "visualizer": {"figsize": [5, 3], "io_window": 1, "default_metric_subsets": ["cpu", "mem", "io"]},
        "reporter": {"thresholds": {}},
        "time_origin": "perf_counter",
        "timezone": "UTC",
    }
    (dir_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_import_session_and_use(ipython, temp_dir):
    # Build fake session
    sess_dir = Path(temp_dir) / "fake_sess"
    _write_fake_session(sess_dir)

    magics = PerfmonitorMagics(ipython, build_perfmonitor_magic_adapter())

    # Import
    magics.import_session(str(sess_dir))

    # Plot should proceed using imported session (we don't actually render)
    called = {"plot": False}
    adapter = magics.magic_adapter
    def _fake_plot(*args, **kwargs):
        called["plot"] = True
    # Patch visualizer.plot
    original_plot = adapter.service.visualizer.plot
    adapter.service.visualizer.plot = _fake_plot
    try:
        magics.perfmonitor_plot("")
        assert called["plot"] is True, "Plot should be invoked for imported session"
    finally:
        adapter.service.visualizer.plot = original_plot

    # Resources should display imported context (no exception)
    magics.perfmonitor_resources("")

