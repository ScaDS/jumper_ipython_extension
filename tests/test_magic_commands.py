from unittest.mock import patch

import pandas as pd

from jumper_extension.magics import (
    load_ipython_extension,
    perfmonitorMagics,
    unload_ipython_extension,
)


def test_initialization_and_basic_operations(ipython, mock_cpu_only):
    """Test initialization, start/stop, and basic operations"""
    magics = perfmonitorMagics(ipython)
    assert magics.monitor is None
    assert not magics.print_perfreports

    # Test start/stop cycle with valid interval parsing
    magics.perfmonitor_start("0.5")
    assert magics.monitor.interval == 0.5
    magics.perfmonitor_stop("")

    # Test already running
    magics.perfmonitor_start("")
    magics.perfmonitor_start("")  # Already running
    magics.perfmonitor_stop("")

    # Test invalid interval (no monitor running)
    magics.perfmonitor_start("invalid")  # Invalid interval


def test_no_monitor_error_cases(ipython, mock_cpu_only):
    """Test commands that require active monitor"""
    magics = perfmonitorMagics(ipython)
    magics.perfmonitor_stop("")
    magics.perfmonitor_resources("")
    magics.perfmonitor_plot("")
    magics.perfmonitor_perfreport("")
    magics.perfmonitor_export_perfdata("")


def test_resources_and_gpu(ipython, mock_cpu_gpu):
    """Test resources display with GPU"""
    magics = perfmonitorMagics(ipython)
    magics.perfmonitor_start("")
    magics.perfmonitor_resources("")
    magics.perfmonitor_stop("")


def test_cell_operations(ipython, mock_cpu_only):
    """Test cell history and reports"""
    magics = perfmonitorMagics(ipython)

    # Test cell history tracking and command
    cell_info = type("Info", (), {"raw_cell": "test"})()
    magics.pre_run_cell(cell_info)
    result = type("Result", (), {"result": None})()
    magics.post_run_cell(result)

    with patch.object(magics.cell_history, "print"):
        perfmonitorMagics.cell_history(magics, "")

    # Test auto-reports
    magics.perfmonitor_start("")
    magics.perfmonitor_enable_perfreports("")
    with patch.object(magics.reporter, "print"):
        magics.post_run_cell(result)
    magics.perfmonitor_disable_perfreports("")

    # Test auto-reports with level option
    magics.perfmonitor_enable_perfreports("--level user")
    assert magics.perfreports_level == "user"
    # First call to post_run_cell resets _skip_report flag
    magics.post_run_cell(result)
    with patch.object(magics.reporter, "display") as mock_display:
        magics.post_run_cell(result)
        # Verify that the reporter.print was called with the correct level
        mock_display.assert_called_with(cell_range=None, level="user")
    magics.perfmonitor_disable_perfreports("")
    magics.perfmonitor_stop("")


def test_plot_scenarios(ipython, mock_cpu_only):
    """Test plotting with different data scenarios"""
    magics = perfmonitorMagics(ipython)
    magics.perfmonitor_start("")

    # Test invalid cell
    magics.perfmonitor_plot("--cell invalid")

    # Test empty data
    with patch.object(
        magics.monitor.data,
        "view",
        return_value=pd.DataFrame(columns=["time"]),
    ):
        magics.perfmonitor_plot("")

    # Test with data
    df = pd.DataFrame({"time": [1.0, 2.0], "cpu_util_avg": [50.0, 60.0]})
    with patch.object(
        magics.monitor.data, "view", return_value=df
    ), patch.object(magics.visualizer, "plot"), patch.object(
        magics.monitor, "start_time", 0.0
    ):
        magics.perfmonitor_plot("")

    # Test with cell filter
    with patch("time.time", side_effect=[1.0, 2.0]):
        cell_info = type("Info", (), {"raw_cell": "test"})()
        magics.pre_run_cell(cell_info)
        magics.post_run_cell(type("Result", (), {"result": None})())

    with patch.object(
        magics.monitor.data, "view", return_value=df
    ), patch.object(magics.visualizer, "plot"), patch.object(
        magics.monitor, "start_time", 0.0
    ):
        magics.perfmonitor_plot("--cell 0")

    magics.perfmonitor_stop("")


def test_perfreport_scenarios(ipython, mock_cpu_only):
    """Test performance reporting scenarios"""
    magics = perfmonitorMagics(ipython)

    # Test no monitor
    magics.perfmonitor_perfreport("")

    magics.perfmonitor_start("")

    # Test invalid cell for perfreport command
    magics.perfmonitor_perfreport("--cell invalid")

    # Add cell to history
    with patch("time.time", side_effect=[1.0, 2.0]):
        cell_info = type("Info", (), {"raw_cell": "test"})()
        magics.pre_run_cell(cell_info)
        magics.post_run_cell(type("Result", (), {"result": None})())

    # Test empty data
    with patch.object(
        magics.monitor.data,
        "view",
        return_value=pd.DataFrame(columns=["time"]),
    ):
        magics.reporter.print()

    # Test with full data
    df = pd.DataFrame(
        {
            "time": [1.0, 2.0],
            "cpu_util_avg": [50.0, 60.0],
            "memory": [4.0, 4.5],
            "gpu_util_avg": [30.0, 40.0],
            "gpu_mem_avg": [2.0, 2.5],
        }
    )
    with patch.object(magics.monitor.data, "view", return_value=df):
        magics.reporter.print()
        magics.reporter.print(
            (0, 0)
        )  # Custom cell marks (use integer indices)
        magics.perfmonitor_perfreport("--cell 0")  # Via command

    # Test with missing columns
    df_partial = pd.DataFrame(
        {
            "time": [1.0, 2.0],
            "cpu_util_avg": [50.0, 60.0],
            "memory": [4.0, 4.5],
        }
    )
    with patch.object(magics.monitor.data, "view", return_value=df_partial):
        magics.reporter.print()

    magics.perfmonitor_stop("")


def test_export_and_help(ipython, mock_cpu_only):
    """Test export functions and help"""
    magics = perfmonitorMagics(ipython)

    # Test exports without monitor
    magics.perfmonitor_export_perfdata("")

    # Test exports with monitor
    magics.perfmonitor_start("")
    with patch.object(magics.monitor.data, "export"):
        magics.perfmonitor_export_perfdata("")
        magics.perfmonitor_export_perfdata("--file custom.csv")
    magics.perfmonitor_stop("")

    # Test cell history export
    with patch.object(magics.cell_history, "export"):
        magics.perfmonitor_export_cell_history("")
        magics.perfmonitor_export_cell_history("--file custom.json")

    # Test CSV export
    with patch.object(magics.cell_history, "export") as mock_export:
        magics.perfmonitor_export_cell_history("--file test.csv")
        mock_export.assert_called_with("test.csv")

    # Test help
    magics.perfmonitor_help("")


def test_extension_lifecycle(ipython, mock_cpu_only):
    """Test IPython extension load/unload"""
    with patch.object(ipython.events, "register"), patch.object(
        ipython, "register_magics"
    ):
        load_ipython_extension(ipython)

    # Test unload with monitor
    from jumper_extension.magics import _perfmonitor_magics

    _perfmonitor_magics.perfmonitor_start("")
    with patch.object(ipython.events, "unregister"):
        unload_ipython_extension(ipython)

    # Test unload without magics
    from jumper_extension import magics

    magics._perfmonitor_magics = None
    unload_ipython_extension(ipython)
