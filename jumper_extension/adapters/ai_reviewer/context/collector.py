import importlib.metadata
import pandas as pd

from jumper_extension.adapters.data import aggregate_node_info
from jumper_extension.adapters.data.node import NodeInfo
from jumper_extension.config.loader import load_config

_EXCLUDED_SUMMARY_COLUMNS = {"time", "cell_index"}

# Context sources the strategy can toggle. Each id maps to the OptimizationState
# field it fills; disabled sources are left empty. Adding a source = add an entry
# here plus a branch in ``_build_sources`` and a ``given`` item with the same id.
_SOURCE_FIELDS = {
    "code": ("cell_code", ""),
    "perf": ("perf_summary", {}),
    "hardware": ("hardware_info", {}),
    "tags": ("perf_tags", []),
    "packages": ("env_info", {}),
}


def collect_env_info(packages: list[str]) -> dict[str, str]:
    """Return ``{package_name: version}`` for each installed package in *packages*.

    Absent packages are omitted entirely, keeping the payload sent to the
    LLM as short as possible.
    """
    result: dict[str, str] = {}
    for pkg in packages:
        try:
            result[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


class ContextCollector:
    """Gathers cell code, performance data and hardware info for the AI review agent.

    Reuses :meth:`PerformanceReporter.build_context`, which already
    assembles cell source code, performance metrics and tags. Which
    sources are actually included is controlled by the strategy's
    ``overrides`` map (``id -> enabled``); disabled sources are left
    empty so both the prompt and the LLM payload stay consistent.
    """

    def __init__(self, reviewer):
        self.reviewer = reviewer

    def collect(self, cell_range=None, level: str = "process", overrides: dict | None = None) -> dict | None:
        overrides = overrides or {}
        ctx = self.reviewer.reporter.build_context(cell_range, level)
        if ctx is None:
            return None

        hardware = aggregate_node_info(self.reviewer.monitor.nodes.hardware)
        sources = self._build_sources(ctx, hardware)

        collected = {"cell_range": ctx["cell_range"]}
        for source_id, (field, empty) in _SOURCE_FIELDS.items():
            enabled = overrides.get(source_id, True)
            collected[field] = sources[source_id] if enabled else empty
        return collected

    def _build_sources(self, ctx: dict, hardware: NodeInfo) -> dict:
        return {
            "code": "\n---\n".join(ctx["filtered_cells"]["raw_cell"]),
            "perf": self._summarize_perfdata(ctx["perfdata"]),
            "hardware": self._hardware_info(hardware),
            "tags": [str(tag_score.tag) for tag_score in ctx["tags_model"]],
            "packages": collect_env_info(load_config().ai.context.known_packages),
        }

    @staticmethod
    def _summarize_perfdata(perfdata: pd.DataFrame) -> dict:
        """Reduce the metrics DataFrame to a {metric: {mean, max}} summary."""
        summary = {}
        for column in perfdata.select_dtypes(include="number").columns:
            if column in _EXCLUDED_SUMMARY_COLUMNS:
                continue
            summary[column] = {
                "mean": float(perfdata[column].mean()),
                "max": float(perfdata[column].max()),
            }
        return summary

    @staticmethod
    def _hardware_info(hardware: NodeInfo) -> dict:
        return {
            "num_cpus": hardware.num_cpus,
            "num_gpus": hardware.num_gpus,
            "gpu_name": hardware.gpu_name,
            "memory_limits": hardware.memory_limits,
        }
