import importlib.metadata
import pandas as pd

from jumper_extension.adapters.data import aggregate_node_info
from jumper_extension.adapters.data.node import NodeInfo
from jumper_extension.config.loader import load_config

_EXCLUDED_SUMMARY_COLUMNS = {"time", "cell_index"}

# Prefixed to each cell's source when a review spans several cells, so per-cell
# durations and metrics can name the block they belong to.
_CELL_MARKER = "# --- cell {index} ---"

# Context sources the strategy can toggle: id -> (state field, empty value,
# default enabled). Disabled sources are left empty and not even built. Adding a
# source = add an entry here plus a builder in ``_build_sources`` and a ``given``
# item with the same id (matching the default here).
_SOURCE_FIELDS = {
    "code": ("cell_code", "", True),
    "timing": ("timing_info", {}, True),
    "perf": ("perf_summary", {}, True),
    "raw_perf": ("raw_perf", {}, False),
    "hardware": ("hardware_info", {}, True),
    "tags": ("perf_tags", [], True),
    "packages": ("env_info", {}, True),
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
        builders = self._build_sources(ctx, hardware)

        # cell_sources never reaches the LLM - apply and the diffs need it - so
        # it ignores the source toggles.
        collected = {
            "cell_range": ctx["cell_range"],
            "cell_sources": self._cell_sources(ctx["filtered_cells"]),
        }
        for source_id, (field, empty, default) in _SOURCE_FIELDS.items():
            enabled = overrides.get(source_id, default)
            collected[field] = builders[source_id]() if enabled else empty
        return collected

    def _build_sources(self, ctx: dict, hardware: NodeInfo) -> dict:
        """Map source id -> zero-arg builder, invoked only when the source is enabled."""
        return {
            "code": lambda: self._cell_code(ctx["filtered_cells"]),
            "timing": lambda: self._timing_info(ctx),
            "perf": lambda: self._summarize_perfdata(ctx["perfdata"]),
            "raw_perf": lambda: ctx["perfdata"].to_dict(orient="list"),
            "hardware": lambda: self._hardware_info(hardware),
            "tags": lambda: [str(tag_score.tag) for tag_score in ctx["tags_model"]],
            "packages": lambda: collect_env_info(load_config().ai.context.known_packages),
        }

    @staticmethod
    def _cell_sources(cells: pd.DataFrame) -> dict[int, str]:
        """Map ``cell_index`` -> that cell's source, verbatim and unmarked."""
        return {
            int(row.cell_index): row.raw_cell
            for row in cells.itertuples(index=False)
        }

    @staticmethod
    def _cell_code(cells: pd.DataFrame) -> str:
        """Join the reviewed cells, each marked with its ``cell_index``.

        A single cell is passed through verbatim: this string is also what
        ``suggest`` rewrites and what the diffs are taken against, so a marker
        there would surface as a line the user never wrote.
        """
        if len(cells) == 1:
            return cells.iloc[0]["raw_cell"]
        return "\n".join(
            f"{_CELL_MARKER.format(index=int(row.cell_index))}\n{row.raw_cell}"
            for row in cells.itertuples(index=False)
        )

    @staticmethod
    def _timing_info(ctx: dict) -> dict:
        """Wall-clock durations of the reviewed cells, per cell and in total.

        Measured by ``CellHistory`` hooks, so - unlike the sampled metrics
        behind ``perf`` - these stay exact for cells too short to sample.
        """
        cells = ctx["filtered_cells"]
        return {
            "total_duration_s": round(float(ctx["total_duration"]), 4),
            "per_cell_duration_s": {
                int(row.cell_index): round(float(row.duration), 4)
                for row in cells.itertuples(index=False)
            },
        }

    @classmethod
    def _summarize_perfdata(cls, perfdata: pd.DataFrame) -> dict:
        """Reduce the metrics DataFrame to an ``overall`` {metric: {mean, max}}
        summary, plus the same per cell.

        Over a range, a single average blends a hot cell with quiet neighbours
        into something unremarkable. ``per_cell`` is omitted for one cell, where
        it would just repeat ``overall``.
        """
        summary = {"overall": cls._summarize_frame(perfdata)}
        if "cell_index" not in perfdata.columns:
            return summary

        groups = perfdata.groupby("cell_index")
        if len(groups) < 2:
            return summary

        summary["per_cell"] = {
            int(cell_index): cls._summarize_frame(group)
            for cell_index, group in groups
        }
        return summary

    @staticmethod
    def _summarize_frame(perfdata: pd.DataFrame) -> dict:
        """Reduce one metrics frame to a {metric: {mean, max}} summary."""
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
