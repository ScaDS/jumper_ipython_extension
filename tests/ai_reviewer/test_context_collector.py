from unittest.mock import Mock

import pandas as pd

from jumper_extension.adapters.data.node import NodeInfo
from jumper_extension.adapters.ai_reviewer.context.collector import ContextCollector


def _make_reviewer(build_context_result):
    reviewer = Mock()
    reviewer.reporter.build_context = Mock(return_value=build_context_result)
    reviewer.monitor.nodes.hardware = {
        "node-0": NodeInfo(
            node="node-0",
            num_cpus=8,
            num_system_cpus=16,
            num_gpus=1,
            gpu_memory=24.0,
            gpu_name="NVIDIA A100",
            memory_limits={"process": 32.0},
        ),
    }
    return reviewer


def _make_context(cell_range=(2, 3)):
    return {
        "cell_range": cell_range,
        "filtered_cells": pd.DataFrame({"raw_cell": ["x = 1", "y = slow(x)"]}),
        "perfdata": pd.DataFrame({
            "time": [1.0, 2.0],
            "cell_index": [2, 3],
            "cpu": [10.0, 90.0],
            "memory": [1.0, 2.0],
        }),
        "tags_model": [
            type("TagScore", (), {"tag": "cpu_bound"})(),
            type("TagScore", (), {"tag": "normal"})(),
        ],
        "total_duration": 5.0,
    }


def test_collect_returns_none_when_no_context_available():
    reviewer = _make_reviewer(build_context_result=None)
    collector = ContextCollector(reviewer)

    assert collector.collect() is None


def test_collect_builds_optimization_state_from_context():
    reviewer = _make_reviewer(build_context_result=_make_context())
    collector = ContextCollector(reviewer)

    state = collector.collect(cell_range=(2, 3), level="user")

    reviewer.reporter.build_context.assert_called_once_with((2, 3), "user")
    assert state["cell_range"] == (2, 3)
    assert state["level"] == "user"
    assert state["cell_code"] == "x = 1\n---\ny = slow(x)"
    assert state["perf_tags"] == ["cpu_bound", "normal"]
    assert state["analysis"] == ""
    assert state["suggestions"] == []
    assert state["chosen_index"] is None
    assert state["applied"] is False


def test_collect_summarizes_perfdata_excluding_time_and_cell_index():
    reviewer = _make_reviewer(build_context_result=_make_context())
    collector = ContextCollector(reviewer)

    state = collector.collect()

    assert set(state["perf_summary"]) == {"cpu", "memory"}
    assert state["perf_summary"]["cpu"] == {"mean": 50.0, "max": 90.0}
    assert state["perf_summary"]["memory"] == {"mean": 1.5, "max": 2.0}


def test_collect_builds_hardware_info_from_aggregated_node_info():
    reviewer = _make_reviewer(build_context_result=_make_context())
    collector = ContextCollector(reviewer)

    state = collector.collect()

    assert state["hardware_info"] == {
        "num_cpus": 8,
        "num_gpus": 1,
        "gpu_name": "NVIDIA A100",
        "memory_limits": {"process": 32.0},
    }
