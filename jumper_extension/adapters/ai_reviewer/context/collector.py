import pandas as pd

from jumper_extension.adapters.data import aggregate_node_info
from jumper_extension.adapters.data.node import NodeInfo
from jumper_extension.adapters.ai_reviewer.agent.state import OptimizationState

_EXCLUDED_SUMMARY_COLUMNS = {"time", "cell_index"}


class ContextCollector:
    """Gathers cell code, performance data and hardware info for the AI review agent.

    Reuses :meth:`PerformanceReporter.build_context`, which already
    assembles cell source code, performance metrics and tags. No shell
    reference is needed here - that is only required by the
    ``apply_suggestion`` node.

    Reads ``reporter``/``monitor`` off the attached :class:`AIReviewer`
    so the live, currently-attached monitor is always used (mirrors how
    :class:`PerformanceReporter`/:class:`PerformanceVisualizer` track
    the monitor through ``attach``).
    """

    def __init__(self, reviewer):
        self.reviewer = reviewer

    def collect(self, cell_range=None, level: str = "process") -> OptimizationState | None:
        ctx = self.reviewer.reporter.build_context(cell_range, level)
        if ctx is None:
            return None

        hardware = aggregate_node_info(self.reviewer.monitor.nodes.hardware)
        cell_code = "\n---\n".join(ctx["filtered_cells"]["raw_cell"])

        return OptimizationState(
            run_id="",
            cell_range=ctx["cell_range"],
            level=level,
            cell_code=cell_code,
            perf_summary=self._summarize_perfdata(ctx["perfdata"]),
            hardware_info=self._hardware_info(hardware),
            perf_tags=[str(tag_score.tag) for tag_score in ctx["tags_model"]],
            analysis="",
            suggestions=[],
            chosen_index=None,
            custom_instruction="",
            refined_code=None,
            applied=False,
        )

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
