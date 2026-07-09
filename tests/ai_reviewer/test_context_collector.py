import importlib.metadata
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from jumper_extension.adapters.data.node import NodeInfo
from jumper_extension.adapters.ai_reviewer.context.collector import (
    ContextCollector,
    collect_env_info,
)


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


def _version_side_effect(pkg):
    installed = {"numpy": "1.26.0", "torch": "2.3.0"}
    if pkg in installed:
        return installed[pkg]
    raise importlib.metadata.PackageNotFoundError(pkg)


# --- collect_env_info ---

def test_collect_env_info_returns_only_installed_packages():
    packages = ["numpy", "torch", "cupy"]

    with patch("importlib.metadata.version", side_effect=_version_side_effect):
        result = collect_env_info(packages)

    assert result == {"numpy": "1.26.0", "torch": "2.3.0"}
    assert "cupy" not in result


def test_collect_env_info_returns_empty_dict_when_nothing_installed():
    with patch("importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError):
        result = collect_env_info(["torch", "jax"])

    assert result == {}


def test_collect_env_info_preserves_input_order():
    packages = ["torch", "numpy", "pandas"]

    with patch("importlib.metadata.version", side_effect=_version_side_effect):
        result = collect_env_info(packages)

    assert list(result.keys()) == ["torch", "numpy"]


# --- ContextCollector ---

def test_collect_returns_none_when_no_context_available():
    reviewer = _make_reviewer(build_context_result=None)
    collector = ContextCollector(reviewer)

    assert collector.collect() is None


def test_collect_builds_optimization_state_from_context():
    reviewer = _make_reviewer(build_context_result=_make_context())
    collector = ContextCollector(reviewer)

    with patch("jumper_extension.adapters.ai_reviewer.context.collector.collect_env_info", return_value={}):
        collected = collector.collect(cell_range=(2, 3), level="user")

    reviewer.reporter.build_context.assert_called_once_with((2, 3), "user")
    assert collected["cell_range"] == (2, 3)
    assert collected["cell_code"] == "x = 1\n---\ny = slow(x)"
    assert collected["perf_tags"] == ["cpu_bound", "normal"]


def test_collect_includes_env_info_in_state():
    reviewer = _make_reviewer(build_context_result=_make_context())
    collector = ContextCollector(reviewer)
    env = {"numpy": "1.26.0", "torch": "2.3.0"}

    with patch("jumper_extension.adapters.ai_reviewer.context.collector.collect_env_info", return_value=env):
        state = collector.collect()

    assert state["env_info"] == env


def test_collect_summarizes_perfdata_excluding_time_and_cell_index():
    reviewer = _make_reviewer(build_context_result=_make_context())
    collector = ContextCollector(reviewer)

    with patch("jumper_extension.adapters.ai_reviewer.context.collector.collect_env_info", return_value={}):
        state = collector.collect()

    assert set(state["perf_summary"]) == {"cpu", "memory"}
    assert state["perf_summary"]["cpu"] == {"mean": 50.0, "max": 90.0}
    assert state["perf_summary"]["memory"] == {"mean": 1.5, "max": 2.0}


def test_collect_builds_hardware_info_from_aggregated_node_info():
    reviewer = _make_reviewer(build_context_result=_make_context())
    collector = ContextCollector(reviewer)

    with patch("jumper_extension.adapters.ai_reviewer.context.collector.collect_env_info", return_value={}):
        state = collector.collect()

    assert state["hardware_info"] == {
        "num_cpus": 8,
        "num_gpus": 1,
        "gpu_name": "NVIDIA A100",
        "memory_limits": {"process": 32.0},
    }
