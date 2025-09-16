import logging

from .extension_messages import (
    ExtensionErrorCode,
    EXTENSION_ERROR_MESSAGES,
)
from typing import List, Tuple, Union

from pathlib import Path
from IPython.display import display, HTML
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .utilities import filter_perfdata
from .analyzer import PerformanceAnalyzer, PerformanceTag, TagScore


logger = logging.getLogger("extension")

class PerformanceReporter:
    def __init__(self, monitor, cell_history, min_duration=None, templates_dir=None):
        self.monitor = monitor
        self.cell_history = cell_history
        self.min_duration = min_duration
        self.analyzer = PerformanceAnalyzer()

        self.templates_dir = Path(templates_dir) if templates_dir else Path(__file__).parent / "templates"

    def print(self, cell_range=None, level="process"):
        """Print performance report"""

        data = self._prepare_report_data(cell_range, level)
        if data is None:
            return

        filtered_cells = data['filtered_cells']
        perfdata = data['perfdata']
        tags_model = data['tags_model']
        total_duration = data['total_duration']

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
        tags_display = self._format_performance_tags(tags_model)
        if tags_display:
            print("Signature(s):")
            tags_line = " | ".join(tag["name"] for tag in tags_display)
            print(tags_line)

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

    def display(self, cell_range=None, level="process"):
        """Print performance report"""

        data = self._prepare_report_data(cell_range, level)
        if data is None:
            return

        filtered_cells = data['filtered_cells']
        perfdata = data['perfdata']
        tags_model = data['tags_model']
        total_duration = data['total_duration']

        tags_display = self._format_performance_tags(tags_model)

        # Build report
        metrics_spec = [
            (f"CPU Util (Across {self.monitor.num_cpus} CPUs)", "cpu_util_avg", "-"),
            ("Memory (GB)", "memory", f"{self.monitor.memory_limits[level]:.2f}" if hasattr(self.monitor, "memory_limits") else "-"),
            (f"GPU Util (Across {getattr(self.monitor, 'num_gpus', 0)} GPUs)", "gpu_util_avg", "-"),
            ("GPU Memory (GB)", "gpu_mem_avg", f"{getattr(self.monitor, 'gpu_memory', 0.0):.2f}"),
        ]
        metrics_rows = []
        for name, col, total in metrics_spec:
            if col in perfdata.columns:
                metrics_rows.append({
                    "name": name,
                    "avg": float(perfdata[col].mean()),
                    "min": float(perfdata[col].min()),
                    "max": float(perfdata[col].max()),
                    "total": total,
                })

        # Render Jinja2 HTML from external files
        env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=select_autoescape(["html", "xml"])
        )
        report_html_path = Path("report") / "report.html"
        template = env.get_template(report_html_path.as_posix())
        # Read external stylesheet and inline it for notebook rendering
        try:
            styles_path = self.templates_dir / "report" / "styles.css"
            inline_styles = styles_path.read_text(encoding="utf-8") if styles_path.exists() else ""
        except Exception:
            inline_styles = ""

        html = template.render(
            duration=total_duration,
            n_cells=len(filtered_cells) if filtered_cells is not None else 1,
            metrics=metrics_rows,
            tags=tags_display,
            inline_styles=inline_styles,
        )
        display(HTML(html))

    def _prepare_report_data(self, cell_range, level):
        """Prepare all necessary data for performance reporting.

        Returns:
            dict: Dictionary containing filtered_cells, perfdata, ranked_tags,
                  total_duration, and other data needed for display methods.
                  Returns None if data preparation fails.
        """

        cell_range = self._resolve_cell_range(cell_range)

        if cell_range is None:
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

        tags_model = self.analyzer.analyze_cell_performance(
            perfdata,
            memory_limit,
            gpu_memory_limit
        )

        # Calculate the total duration of selected cells
        total_duration = filtered_cells["duration"].sum()

        return {
            'cell_range': cell_range,
            'filtered_cells': filtered_cells,
            'perfdata': perfdata,
            'tags_model': tags_model,
            'total_duration': total_duration,
        }

    def _resolve_cell_range(self, cell_range) -> Union[Tuple[int, int], None]:
        """Resolve cell range for performance reporting."""

        if not self.monitor:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_ACTIVE_MONITOR]
            )
            return None

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
                    return cell_range
                else:
                    logger.warning(
                        EXTENSION_ERROR_MESSAGES[
                            ExtensionErrorCode.NO_PERFORMANCE_DATA
                        ]
                    )
                    return None
            else:
                return None

        return cell_range

    @staticmethod
    def _format_performance_tags(ranked_tags: List[TagScore]):
        """Format ranked performance tags for display"""
        if not ranked_tags:
            return [{"name": "UNKNOWN", "slug": "unknown"}]

        # If the only classification is NORMAL, do not display any tag
        if len(ranked_tags) == 1 and ranked_tags[0].tag == PerformanceTag.NORMAL:
            return []

        # Format all tags with their scores/ratios
        tag_displays = []
        for tag_score in ranked_tags:
            # Create slug for CSS hooks and uppercase name for display
            tag_slug = str(tag_score.tag)
            tag_name = tag_slug.upper()
            tag_displays.append({
                "name": tag_name,
                "slug": tag_slug,
            })
        return tag_displays
