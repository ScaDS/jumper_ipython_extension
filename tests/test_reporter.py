from unittest.mock import Mock, patch

import pandas as pd

from jumper_extension.adapters.data.node import NodeInfo
from jumper_extension.adapters.reporter import ReportPrinter


def _make_printer():
    monitor = Mock()
    monitor.nodes.view = Mock(return_value=pd.DataFrame({
        "time": [1.0, 2.0],
        "cpu_util_avg": [10.0, 90.0],
    }))
    monitor.nodes.hardware = {
        "node-0": NodeInfo(
            node="node-0",
            num_cpus=8,
            num_system_cpus=16,
            num_gpus=0,
            gpu_memory=0.0,
            gpu_name="",
            memory_limits={"process": 32.0},
        ),
    }
    cell_history = Mock()
    cell_history.view = Mock(return_value=pd.DataFrame({
        "cell_index": [2, 3],
        "duration": [1.0, 4.0],
    }))
    analyzer = Mock()
    analyzer.analyze_cell_performance = Mock(return_value=[])
    return ReportPrinter(monitor, cell_history, analyzer), monitor, cell_history


def test_prepare_report_data_keeps_idle_samples_and_stays_unlabelled_by_default():
    printer, monitor, _ = _make_printer()

    with patch("jumper_extension.adapters.reporter.filter_perfdata") as filter_mock:
        filter_mock.return_value = pd.DataFrame({"time": [1.0], "cpu_util_avg": [10.0]})
        printer.prepare_report_data((2, 3), "process")

    assert monitor.nodes.view.call_args.kwargs["cell_history"] is None
    assert filter_mock.call_args.kwargs["compress_idle"] is False


def test_prepare_report_data_compresses_idle_and_labels_samples_when_asked():
    printer, monitor, cell_history = _make_printer()

    with patch("jumper_extension.adapters.reporter.filter_perfdata") as filter_mock:
        filter_mock.return_value = pd.DataFrame({"time": [1.0], "cpu_util_avg": [10.0]})
        printer.prepare_report_data(
            (2, 3),
            "process",
            compress_idle=True,
            attach_cell_index=True,
        )

    assert monitor.nodes.view.call_args.kwargs["cell_history"] is cell_history
    assert filter_mock.call_args.kwargs["compress_idle"] is True
