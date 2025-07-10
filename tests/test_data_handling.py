import os

import pandas as pd

from jumper_extension.cell_history import CellHistory
from jumper_extension.performance_data import PerformanceData


def test_performance_data(temp_dir):
    """Test all PerformanceData functionality"""
    # Test initialization and empty dataframe
    data = PerformanceData(num_cpus=2, num_gpus=0)
    assert data.num_cpus == 2 and data.num_gpus == 0 and len(data.data) == 0
    assert len(data.to_dataframe()) == 0

    # Test add_sample and to_dataframe
    data.add_sample(1234567890, [25.0, 30.0], 4.0, [], [], [], [100, 50, 1024, 512])
    assert len(data.data) == 1
    df = data.to_dataframe()
    assert len(df) == 1 and df["cpu_util_avg"].iloc[0] == 27.5

    # Test CSV export
    csv_file = os.path.join(temp_dir, "test.csv")
    data.export(csv_file)
    assert os.path.exists(csv_file) and len(pd.read_csv(csv_file)) == 1


def test_performance_data_gpu():
    """Test GPU functionality and slicing"""
    data = PerformanceData(num_cpus=2, num_gpus=1)
    data.add_sample(
        1234567890, [25.0, 30.0], 4.0, [75.0], [20.0], [60.0], [100, 50, 1024, 512]
    )
    data.add_sample(
        1234567891, [35.0, 40.0], 5.0, [80.0], [25.0], [65.0], [200, 60, 2048, 1024]
    )

    df = data.to_dataframe()
    assert len(df) == 2 and all(
        col in df.columns for col in ["gpu_util_avg", "gpu_band_avg", "gpu_mem_avg"]
    )
    assert len(data.to_dataframe(slice_=(0, 0))) == 1


def test_cell_history(capsys, temp_dir):
    """Test all CellHistory functionality"""
    history = CellHistory()

    # Test start/end cell
    history.start_cell("print('hello')")
    assert history.current_cell["number"] == 0
    history.end_cell(None)
    assert len(history.cells) == 1 and len(history.cell_timestamps) == 1

    # Test print method
    history.print()
    assert "Cell #0" in capsys.readouterr().out

    # Test export
    json_file = os.path.join(temp_dir, "history.json")
    history.export(json_file)
    assert os.path.exists(json_file)
