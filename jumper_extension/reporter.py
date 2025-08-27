import logging

from .extension_messages import (
    ExtensionErrorCode,
    EXTENSION_ERROR_MESSAGES,
)
from typing import List

from .utilities import filter_perfdata
from .analyzer import PerformanceAnalyzer, PerformanceTag


logger = logging.getLogger("extension")


class PerformanceReporter:
    def __init__(self, monitor, cell_history, min_duration=None):
        self.monitor = monitor
        self.cell_history = cell_history
        self.min_duration = min_duration
        self.analyzer = PerformanceAnalyzer()

    @staticmethod
    def _format_performance_tags(ranked_tags: List[TagScore]):
        """Format ranked performance tags for display"""
        if not ranked_tags:
            return "❓ UNKNOWN"

        tag_colors = {
            PerformanceTag.CPU_BOUND: "🔥",
            PerformanceTag.MEMORY_BOUND: "🧠",
            PerformanceTag.GPU_UTIL_BOUND: "🎮",
            PerformanceTag.GPU_MEMORY_BOUND: "💾",
            PerformanceTag.NORMAL: "✅"
        }

        # Handle special cases
        if len(ranked_tags) == 1:
            tag_score = ranked_tags[0]
            if tag_score.tag == PerformanceTag.NORMAL:
                return f"✅ NORMAL"

        # Format all tags with their scores/ratios
        tag_displays = []
        for tag_score in ranked_tags:
            symbol = tag_colors.get(tag_score.tag, "❓")
            # Convert ratio to percentage for display
            percentage = tag_score.score * 100.0
            tag_name = str(tag_score.tag).upper()
            tag_displays.append(f"{symbol} {tag_name} ({percentage:.1f}%)")

        return "\n".join(tag_displays)

    def print(self, cell_range=None, level="process"):
        """Print performance report"""
        if not self.monitor:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_ACTIVE_MONITOR]
            )
            return

        if cell_range is None:
            valid_cells = self.cell_history.view()

            if len(valid_cells) > 0:
                # Filter for non-short cells
                min_duration = (
                    self.min_duration if self.min_duration is not None else 0
                )
                non_short_cells = valid_cells[
                    valid_cells["duration"] >= min_duration
                ]

                if len(non_short_cells) > 0:
                    # Get the last non-short cell index
                    last_valid_cell_idx = int(
                        non_short_cells.iloc[-1]["cell_index"]
                    )
                    cell_range = (last_valid_cell_idx, last_valid_cell_idx)
                else:
                    logger.warning(
                        EXTENSION_ERROR_MESSAGES[
                            ExtensionErrorCode.NO_PERFORMANCE_DATA
                        ]
                    )
                    return
            else:
                return

        # Filter cell history data first using cell_range
        start_idx, end_idx = cell_range
        filtered_cells = self.cell_history.view(start_idx, end_idx + 1)

        perfdata = self.monitor.data.view(level=level)
        perfdata = filter_perfdata(
            filtered_cells, perfdata, compress_idle=False
        )

        # Check if non-empty, otherwise print results
        if perfdata.empty:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.NO_PERFORMANCE_DATA
                ]
            )
            return

        # Analyze cell performance
        memory_limit = self.monitor.memory_limits[level]
        gpu_memory_limit = self.monitor.gpu_memory if self.monitor.num_gpus > 0 else None

        ranked_tags = self.analyzer.analyze_cell_performance(
            perfdata,
            memory_limit,
            gpu_memory_limit
        )

        # Calculate the total duration of selected cells
        total_duration = filtered_cells["duration"].sum()

        print("-" * 40)
        print("JUmPER Performance Report")
        print("-" * 40)
        n_cells = len(filtered_cells)
        print(
            f"Duration: {total_duration:.2f}s "
            f"({n_cells} cell{'s' if n_cells != 1 else ''})"
        )
        print("-" * 40)

        # Output performance tags
        tags_display = self._format_performance_tags(ranked_tags)
        print(f"Performance Tags:\n{tags_display}")

        # Report table
        metrics = [
            (
                f"CPU Util (Across {self.monitor.num_cpus} CPUs)",
                "cpu_util_avg",
                "-",
            ),
            (
                "Memory (GB)",
                "memory",
                f"{self.monitor.memory_limits[level]:.2f}",
            ),
            (
                f"GPU Util (Across {self.monitor.num_gpus} GPUs)",
                "gpu_util_avg",
                "-",
            ),
            (
                "GPU Memory (GB)",
                "gpu_mem_avg",
                f"{self.monitor.gpu_memory:.2f}",
            ),
        ]

        print(f"{'Metric':<25} {'AVG':<8} {'MIN':<8} {'MAX':<8} {'TOTAL':<8}")
        print("-" * 65)
        for name, col, total in metrics:
            if col in perfdata.columns:
                print(
                    f"{name:<25} {perfdata[col].mean():<8.2f} "
                    f"{perfdata[col].min():<8.2f} {perfdata[col].max():<8.2f} "
                    f"{total:<8}"
                )
