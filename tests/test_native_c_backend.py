import os
import shutil
import time

import pytest

from jumper_extension.monitor.backends.native_c.build import (
    _BINARY_PATH,
    build_collector,
)
from jumper_extension.monitor.backends.native_c import CSubprocessPerformanceMonitor
from jumper_extension.config.collectors.c import load_c_collectors_config

pytestmark = pytest.mark.skipif(
    not shutil.which("make"),
    reason="make not available",
)

_COLLECTOR_COLUMNS = {
    "cpu": lambda n_cpus, _: (
        ["cpu_util_avg", "cpu_util_min", "cpu_util_max"]
        + [f"cpu_util_{i}" for i in range(n_cpus)]
    ),
    "memory": lambda *_: ["memory"],
    "io": lambda *_: ["io_read_count", "io_write_count", "io_read", "io_write"],
    "gpu": lambda _, n_gpus: (
        [f"gpu_{m}_{s}" for m in ("util", "band", "mem")
                        for s in ("avg", "min", "max") + tuple(range(n_gpus))]
        if n_gpus > 0 else []
    ),
}


def _expected_columns(active: list[str], num_cpus: int, num_gpus: int) -> list[str]:
    cols = ["time"]
    for name in active:
        cols += _COLLECTOR_COLUMNS[name](num_cpus, num_gpus)
    return cols


def test_build():
    """build_collector() compiles the binary via make."""
    assert build_collector(), "build_collector() returned False"
    assert os.path.isfile(_BINARY_PATH), f"Binary not found at {_BINARY_PATH}"


def test_start_stop():
    """Monitor starts, sets running=True, then stops cleanly."""
    monitor = CSubprocessPerformanceMonitor()
    assert not monitor.running

    monitor.start(interval=1.0)
    assert monitor.running

    monitor.stop()
    assert not monitor.running


def test_collects_data():
    """Monitor collects >=2 samples per level with all protocol columns and no NaN."""
    monitor = CSubprocessPerformanceMonitor()
    monitor.start(interval=1.0)
    time.sleep(3.5)
    monitor.stop()

    hw = monitor.nodes.hardware["local"]
    active = load_c_collectors_config()
    expected = _expected_columns(active, hw.num_system_cpus, hw.num_gpus)

    for level in monitor.nodes.levels:
        df = monitor.nodes.view(level=level)
        assert len(df) >= 2, f"level '{level}': expected >=2 rows, got {len(df)}"
        for col in expected:
            assert col in df.columns, f"level '{level}': missing column '{col}'"
            assert not df[col].isna().any(), f"level '{level}': NaN in '{col}'"
