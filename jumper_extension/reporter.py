import logging

from .extension_messages import (
    ExtensionErrorCode,
    EXTENSION_ERROR_MESSAGES,
)
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
    def _format_performance_tags(profile):
        """Format performance tags for display"""
        tag_colors = {
            PerformanceTag.CPU_BOUND: "",
            PerformanceTag.MEMORY_BOUND: "",
            PerformanceTag.GPU_BOUND: "",
            PerformanceTag.IO_BOUND: "",
            PerformanceTag.BALANCED: "",
            PerformanceTag.IDLE: "⚪"
        }

        primary_color = tag_colors.get(profile.primary_tag, "❓")
        primary_text = f"{primary_color} {profile.primary_tag.value.upper()}"

        if profile.secondary_tags:
            secondary_text = " + " + " + ".join([
                f"{tag_colors.get(tag, '❓')}{tag.value}"
                for tag in profile.secondary_tags
            ])
            return f"{primary_text}{secondary_text}"

        return primary_text

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

        print(f'{perfdata=}')
        input()
        performance_profile = self.analyzer.analyze_cell_performance(
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

        # Output performance profile
        tags_display = self._format_performance_tags(performance_profile)
        print(f"Performance Profile: {tags_display} "
              f"(confidence: {performance_profile.confidence:.2f})")

        # Bottleneck details
        if performance_profile.bottleneck_score:
            print("Bottleneck Analysis:")
            for resource, score in sorted(
                    performance_profile.bottleneck_score.items(),
                    key=lambda x: x[1],
                    reverse=True
            ):
                if score > 10:  # Show only bottlenecks above threshold
                    print(f"  {resource.upper()}: {score:.1f}%")

        print("-" * 40)

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
