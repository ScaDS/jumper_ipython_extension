import os
from unittest.mock import patch

import pytest
import pandas as pd

from jumper_extension.cell_history import CellHistory
from jumper_extension.data import PerformanceData


def test_performance_data(temp_dir):
    """Test PerformanceData functionality"""
    # Test initialization and empty dataframe
    data = PerformanceData(num_cpus=2, num_gpus=0)
    assert data.num_cpus == 2 and data.num_gpus == 0 and len(data.data) == 0
    assert len(data.view()) == 0

    # Test add_sample and view
    data.add_sample(1234567890, [25.0, 30.0], 4.0, [], [], [], [100, 50, 1024, 512])
    assert len(data.data) == 1
    df = data.view()
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

    df = data.view()
    assert len(df) == 2 and all(
        col in df.columns for col in ["gpu_util_avg", "gpu_band_avg", "gpu_mem_avg"]
    )
    assert len(data.view(slice_=(0, 0))) == 1


# === Test CellHistory functionality ===
@pytest.fixture
def simple_history():
    history = CellHistory()
    history.start_cell("print('hello')")
    history.end_cell(None)
    return history


def test_start_current_end_cell():
    history = CellHistory()
    history.start_cell("print('hello')")
    assert history.current_cell["index"] == 0
    history.end_cell(None)
    assert len(history.data) == 1


def test_view_method(simple_history, capsys):
    df = simple_history.view()
    assert len(df) == 1
    assert df.iloc[0]["index"] == 0
    assert df.iloc[0]["raw_cell"] == "print('hello')"
    assert df.iloc[0]["start_time"] < df.iloc[0]["end_time"]

    # Test print method
    simple_history.print()
    assert "Cell #0" in capsys.readouterr().out


def test_show_itable(simple_history):
    with patch("jumper_extension.cell_history.show") as mock_show:
        simple_history.show_itable()
        assert mock_show.called, "Expected show() to be called"

        df_arg = mock_show.call_args[0][0]  # Get pd.DataFrame
        assert isinstance(df_arg, pd.DataFrame)
        assert "Code" in df_arg.columns
        assert df_arg.loc[0, "Code"] == "print('hello')"


def test_export_method(simple_history, tmp_path):
    json_file = tmp_path / "history.json"
    simple_history.export(str(json_file))
    assert json_file.exists()


def test_csv_export_functionality(simple_history, temp_dir):
    csv_file = os.path.join(temp_dir, "history.csv")
    simple_history.export(csv_file)
    assert os.path.exists(csv_file)


def test_view_operations(simple_history):
    assert not simple_history.data.empty
    assert "start_time" in simple_history.data.columns
    assert "end_time" in simple_history.data.columns
    assert "duration" in simple_history.data.columns
    assert "raw_cell" in simple_history.data.columns
    assert "index" in simple_history.data.columns


def test_is_duration_calculated_correctly(simple_history):
    df = simple_history.view()
    assert df.iloc[0]["duration"] == df.iloc[0]["end_time"] - df.iloc[0]["start_time"]
