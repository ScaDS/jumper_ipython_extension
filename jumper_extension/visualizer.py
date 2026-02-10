import logging
import re
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from IPython.display import display
from ipywidgets import widgets, Layout

from .extension_messages import (
    ExtensionErrorCode,
    EXTENSION_ERROR_MESSAGES,
)
from .utilities import (
    filter_perfdata, 
    get_available_levels,
    load_perfdata_from_disk,
    load_monitor_metadata_from_disk
)
from .logo import logo_image, jumper_colors
from .bali_adapter import BaliVisualizationMixin

logger = logging.getLogger("extension")


def is_ipympl_backend():
    try:
        backend = plt.get_backend().lower()
    except Exception:
        return False
    return ("ipympl" in backend) or ("widget" in backend)


class PerformanceVisualizer(BaliVisualizationMixin):
    """Visualizes performance metrics collected by PerformanceMonitor.

    Supports multiple levels: 'user', 'process' (default), 'system', and
    'slurm' (if available)
    """

    def __init__(self, monitor, cell_history, min_duration=None, bali_adapter=None):
        self.monitor = monitor
        self.cell_history = cell_history
        self.figsize = (5, 3)
        self.min_duration = min_duration
        # Smooth IO with ~1s rolling window based on sampling interval
        try:
            self._io_window = max(
                1, int(round(1.0 / (self.monitor.interval or 1.0)))
            )
        except Exception:
            self._io_window = 1

        # Initialize BALI functionality via mixin
        super().__init__(bali_adapter=bali_adapter)

        # Compressed metrics configuration (dict-based entries for clarity)
        self.subsets = {
            "cpu_all": {
                "cpu": {
                    "type": "multi_series",
                    "prefix": "cpu_util_",
                    "title": "CPU Utilization (%) - Across Cores",
                    "ylim": (0, 100),
                    "label": "CPU Utilization (All Cores)",
                }
            },
            "gpu_all": {
                "gpu_util": {
                    "type": "multi_series",
                    "prefix": "gpu_util_",
                    "title": "GPU Utilization (%) - Across GPUs",
                    "ylim": (0, 100),
                    "label": "GPU Utilization (All GPUs)",
                },
                "gpu_band": {
                    "type": "multi_series",
                    "prefix": "gpu_band_",
                    "title": "GPU Bandwidth Usage (%) - Across GPUs",
                    "ylim": (0, 100),
                    "label": "GPU Bandwidth (All GPUs)",
                },
                "gpu_mem": {
                    "type": "multi_series",
                    "prefix": "gpu_mem_",
                    "title": "GPU Memory Usage (GB) - Across GPUs",
                    "ylim": (0, monitor.gpu_memory),
                    "label": "GPU Memory (All GPUs)",
                },
                "gpu_power": {
                    "type": "multi_series",
                    "prefix": "gpu_power_",
                    "title": "GPU Power Usage (W) - Across GPUs",
                    "ylim": None,
                    "label": "GPU Power (All GPUs)",
                },
            },
            "cpu": {
                "cpu_summary": {
                    "type": "summary_series",
                    "columns": [
                        "cpu_util_min",
                        "cpu_util_avg",
                        "cpu_util_max",
                    ],
                    "title": (
                        "CPU Utilization (%) - "
                        f"{self.monitor.num_cpus} CPUs"
                    ),
                    "ylim": (0, 100),
                    "label": "CPU Utilization Summary",
                }
            },
            "gpu": {
                "gpu_util_summary": {
                    "type": "summary_series",
                    "columns": [
                        "gpu_util_min",
                        "gpu_util_avg",
                        "gpu_util_max",
                    ],
                    "title": (
                        "GPU Utilization (%) - "
                        f"{self.monitor.num_gpus} GPUs"
                    ),
                    "ylim": (0, 100),
                    "label": "GPU Utilization Summary",
                },
                "gpu_band_summary": {
                    "type": "summary_series",
                    "columns": [
                        "gpu_band_min",
                        "gpu_band_avg",
                        "gpu_band_max",
                    ],
                    "title": (
                        "GPU Bandwidth Usage (%) - "
                        f"{self.monitor.num_gpus} GPUs"
                    ),
                    "ylim": (0, 100),
                    "label": "GPU Bandwidth Summary",
                },
                "gpu_mem_summary": {
                    "type": "summary_series",
                    "columns": ["gpu_mem_min", "gpu_mem_avg", "gpu_mem_max"],
                    "title": (
                        "GPU Memory Usage (GB) - "
                        f"{self.monitor.num_gpus} GPUs"
                    ),
                    "ylim": (0, monitor.gpu_memory),
                    "label": "GPU Memory Summary",
                },
                "gpu_power_summary": {
                    "type": "summary_series",
                    "columns": [
                        "gpu_power_min",
                        "gpu_power_avg",
                        "gpu_power_max",
                    ],
                    "title": (
                        "GPU Power Usage (W) - "
                        f"{self.monitor.num_gpus} GPUs"
                    ),
                    "ylim": None,
                    "label": "GPU Power Summary",
                },
            },
            "mem": {
                "memory": {
                    "type": "single_series",
                    "column": "memory",
                    "title": "Memory Usage (GB)",
                    "ylim": None,  # Will be set dynamically based on level
                    "label": "Memory Usage",
                }
            },
            "io": {
                "io_read": {
                    "type": "single_series",
                    "column": "io_read",
                    "title": "I/O Read (MB/s)",
                    "ylim": None,
                    "label": "IO Read MB/s",
                },
                "io_write": {
                    "type": "single_series",
                    "column": "io_write",
                    "title": "I/O Write (MB/s)",
                    "ylim": None,
                    "label": "IO Write MB/s",
                },
                "io_read_count": {
                    "type": "single_series",
                    "column": "io_read_count",
                    "title": "I/O Read Operations (ops/s)",
                    "ylim": None,
                    "label": "IO Read Ops",
                },
                "io_write_count": {
                    "type": "single_series",
                    "column": "io_write_count",
                    "title": "I/O Write Operations (ops/s)",
                    "ylim": None,
                    "label": "IO Write Ops",
                },
            },
        }

    def _compress_time_axis(self, perfdata, cell_range):
        """Compress time axis by removing idle periods between cells"""
        if perfdata.empty:
            return perfdata, []

        start_idx, end_idx = cell_range
        cell_data = self.cell_history.view(start_idx, end_idx + 1)
        compressed_perfdata, cell_boundaries, current_time = (
            perfdata.copy(),
            [],
            0,
        )

        for idx, cell in cell_data.iterrows():
            cell_mask = (perfdata["time"] >= cell["start_time"]) & (
                perfdata["time"] <= cell["end_time"]
            )
            cell_perfdata = perfdata[cell_mask]

            if not cell_perfdata.empty:
                original_start, cell_duration = (
                    cell["start_time"],
                    cell["end_time"] - cell["start_time"],
                )
                compressed_perfdata.loc[cell_mask, "time"] = current_time + (
                    cell_perfdata["time"].values - original_start
                )
                cell_boundaries.append(
                    {
                        "index": cell["index"],
                        "start_time": current_time,
                        "end_time": current_time + cell_duration,
                        "duration": cell_duration,
                    }
                )
                current_time += cell_duration

        return compressed_perfdata, cell_boundaries

    def _plot_metric(
        self,
        df,
        metric,
        cell_range=None,
        show_idle=False,
        ax: plt.Axes = None,
        level="process",
        show_bali=False,
        custom_vmin_vmax=None,
    ):
        """Plot a single metric using its configuration"""
        config = next(
            (
                subset[metric]
                for subset in self.subsets.values()
                if metric in subset
            ),
            None,
        )
        if not config:
            return

        # Parse dict-based config format
        if not isinstance(config, dict):
            return

        plot_type = config.get("type")
        if plot_type == "single_series":
            column = config.get("column")
            title = config.get("title", "")
            ylim = config.get("ylim")
            # Set dynamic memory limit for memory metric
            if metric == "memory" and ylim is None:
                ylim = (0, self.monitor.memory_limits[level])
            if not column or column not in df.columns:
                return
        elif plot_type == "multi_series":
            prefix = config.get("prefix", "")
            title = config.get("title", "")
            ylim = config.get("ylim")
            series_cols = [
                col
                for col in df.columns
                if prefix
                and col.startswith(prefix)
                and not col.endswith("avg")
            ]
            # Derive average column name from prefix
            avg_column = f"{prefix}avg" if prefix else None
            if (
                avg_column is None or avg_column not in df.columns
            ) and not series_cols:
                return
        elif plot_type == "summary_series":
            columns = config.get("columns", [])
            title = config.get("title", "")
            ylim = config.get("ylim")
            if level == "system":
                title = re.sub(
                    r"\d+", str(self.monitor.num_system_cpus), title
                )
            available_cols = [col for col in columns if col in df.columns]
            if not available_cols:
                return
        else:
            return

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        # Plot based on type
        if plot_type == "single_series":
            series = df[column]
            # For IO metrics, compute simple diffs from cumulative counters
            if metric in (
                "io_read",
                "io_write",
                "io_read_count",
                "io_write_count",
            ):
                diffs = df[column].astype(float).diff().clip(lower=0)
                if metric in ("io_read", "io_write"):
                    diffs = diffs / (1024**2)  # bytes -> MB
                series = diffs.fillna(0.0)
                if self._io_window > 1:
                    series = series.rolling(
                        window=self._io_window, min_periods=1
                    ).mean()

            ax.plot(df["time"], series, color="blue", linewidth=2)
        elif plot_type == "summary_series":
            line_styles, alpha_vals = ["dotted", "-", "--"], [0.35, 1.0, 0.35]
            for i, (col, label) in enumerate(
                zip(columns, ["Min", "Average", "Max"])
            ):
                if col in df.columns:
                    ax.plot(
                        df["time"],
                        df[col],
                        color="blue",
                        linestyle=line_styles[i % len(line_styles)],
                        linewidth=2,
                        alpha=alpha_vals[i % len(alpha_vals)],
                        label=label,
                    )
            ax.legend()
        elif plot_type == "multi_series":
            for col in series_cols:
                ax.plot(df["time"], df[col], "-", alpha=0.5, label=col)
            if avg_column in df.columns:
                ax.plot(
                    df["time"], df[avg_column], "b-", linewidth=2, label="Mean"
                )
            ax.legend()

        # Apply settings
        ax.set_title(title + (" (No Idle)" if not show_idle else ""))
        ax.set_xlabel("Time (seconds)")
        ax.grid(True)
        if ylim:
            ax.set_ylim(ylim)
        if not show_bali:
            self._draw_cell_boundaries(ax, cell_range, show_idle)
        self._draw_bali_segments(ax, show_bali, show_idle, cell_range, custom_vmin_vmax)

    def _draw_cell_boundaries(self, ax, cell_range=None, show_idle=False):
        """Draw cell boundaries as colored rectangles with cell indices"""
        colors = jumper_colors
        y_min, y_max = ax.get_ylim()
        x_max, height = ax.get_xlim()[1], y_max - y_min
        min_duration = self.min_duration or 0

        def draw_cell_rect(start_time, duration, cell_num, alpha):
            if (
                duration < min_duration
                or start_time > x_max
                or start_time + duration < 0
            ):
                return
            color = colors[cell_num % len(colors)]
            ax.add_patch(
                plt.Rectangle(
                    (start_time, y_min),
                    duration,
                    height,
                    facecolor=color,
                    alpha=alpha,
                    edgecolor="gray",
                    linestyle="--",
                    linewidth=1,
                    zorder=0,
                )
            )
            ax.text(
                start_time + duration / 2,
                y_max - height * 0.1,
                f"#{cell_num}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                zorder=1,
                bbox=dict(
                    boxstyle="round,pad=0.3", facecolor="white", alpha=0.8
                ),
            )

        if not show_idle and hasattr(self, "_compressed_cell_boundaries"):
            for cell in self._compressed_cell_boundaries:
                draw_cell_rect(
                    cell["start_time"],
                    cell["duration"],
                    int(cell["index"]),
                    0.4,
                )
        else:
            filtered_cells = self.cell_history.view()
            cells = (
                filtered_cells.iloc[cell_range[0] : cell_range[1] + 1]
                if cell_range
                else filtered_cells
            )
            for idx, cell in cells.iterrows():
                start_time = cell["start_time"] - self.monitor.start_time
                draw_cell_rect(
                    start_time, cell["duration"], int(cell["index"]), 0.5
                )

    def _draw_bali_segments(
        self, ax, show_bali=False, show_idle=True, cell_range=None, custom_vmin_vmax=None
    ):
        """Draw BALI segments as colored rectangles with click selection."""
        if not show_bali:
            return

        segments = self._load_bali_segments()
        if not segments:
            return

        y_min, y_max = ax.get_ylim()
        x_max, height = ax.get_xlim()[1], y_max - y_min

        # Use compressed or adjusted segments
        if not show_idle and hasattr(self, "_compressed_bali_segments"):
            draw_segments = self._compressed_bali_segments
        else:
            draw_segments = [
                {**s, "start_time": s["start_time"] - self.monitor.start_time}
                for s in segments
            ]

        # Use custom vmin/vmax if provided, otherwise use data range
        if custom_vmin_vmax:
            vmin, vmax = custom_vmin_vmax
        else:
            vmin, vmax = self.bali_adapter.get_tokens_per_sec_range(segments)

        # Clean up previous event handlers
        if hasattr(ax, "_bali_click"):
            ax.figure.canvas.mpl_disconnect(ax._bali_click)

        ax._bali_patches = []
        ax._bali_selected_patch = None

        # Draw segments
        for s in draw_segments:
            start, dur, tps, is_error = (
                s["start_time"],
                s["duration"],
                s["tokens_per_sec"],
                s.get("is_error", False),
            )
            if start > x_max or start + dur < 0:
                continue

            if is_error or tps is None:
                color = "none"
                hatch = None
                edgecolor = "gray"
                alpha = 1.0
                is_error_segment = True
            else:
                # Normal segments: colored based on tokens/sec
                color = self.bali_adapter.get_color_for_tokens_per_sec(
                    tps, vmin, vmax
                )
                hatch = None
                edgecolor = "gray"
                alpha = 0.75
                is_error_segment = False
            
            rect = plt.Rectangle(
                (start, y_min),
                dur,
                height,
                facecolor=color,
                alpha=alpha,
                edgecolor=edgecolor,
                linestyle="--",
                linewidth=1,
                hatch=hatch,
                zorder=0.5,
            )
            
            # Explicitly set edge color to ensure matplotlib state doesn't interfere
            rect.set_edgecolor(edgecolor)
            rect._bali_info = {
                "model": s.get("model", "n/a"),
                "framework": s.get("framework", "n/a"),
                "batch_size": s.get("batch_size", "n/a"),
                "input_len": s.get("input_len", "n/a"),
                "output_len": s.get("output_len", "n/a"),
                "tokens_per_sec": f"{tps:.2f}" if tps else "NaN",
                "duration": f"{s.get('duration', 0):.2f}",
                "is_error": is_error,
                "error_message": s.get("error_message", ""),
            }
            # Store original style for restoration after selection
            rect._original_style = {
                "facecolor": color,
                "edgecolor": edgecolor,
                "hatch": hatch,
                "linewidth": 1,
                "is_error_segment": is_error_segment,
            }
            ax.add_patch(rect)
            ax._bali_patches.append(rect)

        def on_click(event):
            if event.inaxes != ax:
                return
            for patch in ax._bali_patches:
                if patch.contains(event)[0]:
                    # Reset previous selection to original style
                    if ax._bali_selected_patch:
                        prev_patch = ax._bali_selected_patch
                        original = prev_patch._original_style
                        prev_patch.set_hatch(original["hatch"])
                        prev_patch.set_linewidth(original["linewidth"])
                        prev_patch.set_edgecolor(original["edgecolor"])
                        prev_patch.set_facecolor(original["facecolor"])

                    # Highlight new selection
                    patch.set_hatch("///")
                    patch.set_linewidth(1.5)
                    patch.set_edgecolor("black")
                    ax._bali_selected_patch = patch

                    # Print details
                    if hasattr(ax, "_bali_selection_output"):
                        with ax._bali_selection_output:
                            ax._bali_selection_output.clear_output(wait=True)
                            info = patch._bali_info
                            
                            if info.get("is_error", False):
                                print("Failed segment")
                                print(
                                    f"""BALI segment selected:
- Model: {info['model']}
- Framework: {info['framework']}
- Batch size: {info['batch_size']}
- Input len: {info['input_len']}
- Output len: {info['output_len']}
- Error: {info['error_message']}
- Duration (s): {info['duration']}"""
                                )
                            else:
                                print(
                                    f"""BALI segment selected:
- Model: {info['model']}
- Framework: {info['framework']}
- Batch size: {info['batch_size']}
- Input len: {info['input_len']}
- Output len: {info['output_len']}
- Tokens/sec: {info['tokens_per_sec']}
- Duration (s): {info['duration']}"""
                                )

                    ax.figure.canvas.draw_idle()
                    return

        ax._bali_click = ax.figure.canvas.mpl_connect(
            "button_press_event", on_click
        )

    def plot(
        self,
        metric_subsets=("cpu", "mem", "io"),
        cell_range=None,
        show_idle=False,
    ):
        if self.monitor.num_gpus:
            metric_subsets += (
                "gpu",
                "gpu_all",
            )

        """Plot performance metrics with interactive widgets for
        configuration."""
        valid_cells = self.cell_history.view()
        if len(valid_cells) == 0:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_CELL_HISTORY]
            )
            return

        # Default to all cells if no range specified
        min_cell_idx, max_cell_idx = int(valid_cells.iloc[0]["index"]), int(
            valid_cells.iloc[-1]["index"]
        )
        if cell_range is None:
            cell_start_index = 0
            for cell_idx in range(len(valid_cells) - 1, -1, -1):
                if valid_cells.iloc[cell_idx]["duration"] > self.min_duration:
                    cell_start_index = cell_idx
                    break
            cell_range = (
                int(valid_cells.iloc[cell_start_index]["index"]),
                int(valid_cells.iloc[-1]["index"]),
            )

        # Create interactive widgets
        style = {"description_width": "initial"}
        show_idle_checkbox = widgets.Checkbox(
            value=show_idle, description="Show idle periods"
        )
        show_bali_checkbox = widgets.Checkbox(
            value=False, description="Show BALI segments"
        )
        cell_range_slider = widgets.IntRangeSlider(
            value=cell_range,
            min=min_cell_idx,
            max=max_cell_idx,
            step=1,
            description="Cell range:",
            style=style,
        )

        logo_widget = widgets.HTML(
            value=f"<img src="
            f'"{logo_image}"'
            f'alt="JUmPER Logo" style="height: auto; width: 100px;">'
        )

        box_layout = Layout(
            display="flex",
            flex_flow="row wrap",
            align_items="center",
            justify_content="space-between",
            width="100%",
        )

        config_widgets = widgets.HBox(
            [
                widgets.HTML("<b>Plot Configuration:</b>"),
                show_idle_checkbox,
                show_bali_checkbox,
                cell_range_slider,
                logo_widget,
            ],
            layout=box_layout,
        )
        plot_output = widgets.Output()

        # Store the plot wrapper instance for persistent updates
        plot_wrapper = None

        def update_plots():
            nonlocal plot_wrapper
            current_cell_range, current_show_idle, current_show_bali = (
                cell_range_slider.value,
                show_idle_checkbox.value,
                show_bali_checkbox.value,
            )
            start_idx, end_idx = current_cell_range
            filtered_cells = self.cell_history.view(start_idx, end_idx + 1)
            # Store all level data for subplot access
            perfdata_by_level = {}
            for available_level in get_available_levels():
                perfdata_by_level[available_level] = filter_perfdata(
                    filtered_cells,
                    self.monitor.data.view(level=available_level),
                    not current_show_idle,
                )

            if all(df.empty for df in perfdata_by_level.values()):
                with plot_output:
                    plot_output.clear_output()
                    logger.warning(
                        EXTENSION_ERROR_MESSAGES[
                            ExtensionErrorCode.NO_PERFORMANCE_DATA
                        ]
                    )
                    # Clear plot wrapper when no data
                    plot_wrapper = None
                return

            # Handle time compression or show idle for all levels
            processed_perfdata = {}
            for level_key, perfdata in perfdata_by_level.items():
                if not perfdata.empty:
                    if not current_show_idle:
                        processed_data, self._compressed_cell_boundaries = (
                            self._compress_time_axis(
                                perfdata, current_cell_range
                            )
                        )
                        processed_perfdata[level_key] = processed_data
                    else:
                        processed_data = perfdata.copy()
                        processed_data["time"] -= self.monitor.start_time
                        processed_perfdata[level_key] = processed_data
                else:
                    processed_perfdata[level_key] = perfdata

            # Handle BALI segments compression
            if current_show_bali and not current_show_idle:
                bali_segments = self._load_bali_segments()
                primary_level = get_available_levels()[0]
                reference_perfdata = perfdata_by_level.get(primary_level)
                self._compressed_bali_segments = (
                    self.bali_adapter.compress_segments(
                        bali_segments, current_cell_range, reference_perfdata, 
                        self.cell_history
                    )
                    if reference_perfdata is not None
                    and not reference_perfdata.empty
                    else []
                )

            # Get metrics for subsets and build labeled dropdown options
            metrics = []
            labeled_options = []
            for subset in metric_subsets:
                if subset in self.subsets:
                    for metric_key, cfg in self.subsets[subset].items():
                        metrics.append(metric_key)
                        label = (
                            cfg.get("label")
                            if isinstance(cfg, dict)
                            else metric_key
                        )
                        labeled_options.append(
                            (label or metric_key, metric_key)
                        )
                else:
                    logger.warning(
                        EXTENSION_ERROR_MESSAGES[
                            ExtensionErrorCode.INVALID_METRIC_SUBSET
                        ].format(
                            subset=subset,
                            supported_subsets=", ".join(self.subsets.keys()),
                        )
                    )

            with plot_output:
                # Recreate wrapper if needed or BALI state changed
                if (
                    plot_wrapper is None
                    or getattr(plot_wrapper, "show_bali", None)
                    != current_show_bali
                ):
                    plot_output.clear_output()
                    plot_wrapper = InteractivePlotWrapper(
                        self._plot_metric,
                        metrics,
                        labeled_options,
                        processed_perfdata,
                        current_cell_range,
                        current_show_idle,
                        current_show_bali,
                        self.figsize,
                    )
                    plot_wrapper.monitor = self.monitor
                    plot_wrapper.display_ui()
                else:
                    plot_wrapper.update_data(
                        processed_perfdata,
                        current_cell_range,
                        current_show_idle,
                    )

        # Set up observers and display
        for widget in [
            show_idle_checkbox,
            show_bali_checkbox,
            cell_range_slider,
        ]:
            widget.observe(lambda change: update_plots(), names="value")

        display(widgets.VBox([config_widgets, plot_output]))
        update_plots()


class InteractivePlotWrapper:
    """Interactive plotter with dropdown selection and reusable matplotlib
    axes."""

    def __init__(
        self,
        plot_callback,
        metrics: List[str],
        labeled_options,
        perfdata_by_level,
        cell_range=None,
        show_idle=False,
        show_bali=False,
        figsize=None,
    ):
        self.plot_callback, self.perfdata_by_level, self.metrics = (
            plot_callback,
            perfdata_by_level,
            metrics,
        )
        self.labeled_options = labeled_options
        self.cell_range, self.show_idle, self.show_bali, self.figsize = (
            cell_range,
            show_idle,
            show_bali,
            figsize,
        )
        self.shown_metrics, self.panel_count, self.max_panels = (
            set(),
            0,
            len(metrics) * 4,
        )
        # Store plot panels for updates
        self.plot_panels = []
        # Initialize custom_vmin_vmax for all instances
        self.custom_vmin_vmax = None

        self.output_container = widgets.HBox(
            layout=Layout(
                display="flex",
                flex_flow="row wrap",
                align_items="center",
                justify_content="space-between",
                width="100%",
            )
        )
        self.add_panel_button = widgets.Button(
            description="Add Plot Panel",
            layout=Layout(margin="0 auto 20px auto"),
        )
        self.add_panel_button.on_click(self._on_add_panel_clicked)

        # BALI colorbar components
        if show_bali:
            self.bali_colorbar_output = widgets.Output()
            # Add vmin/vmax controls
            self.vmin_widget = widgets.FloatText(value=0.0, description="vmin:", step=0.1)
            self.vmax_widget = widgets.FloatText(value=100.0, description="vmax:", step=0.1)
            self.bali_colorbar_container = widgets.HBox(
                [self.vmin_widget, self.vmax_widget, self.bali_colorbar_output],
                layout=Layout(
                    display="flex", justify_content="center", width="100%"
                ),
            )
            # Use the same BALI adapter from the parent visualizer
            self.bali_adapter = getattr(
                self.plot_callback.__self__, "bali_adapter", None
            )
        else:
            self.bali_colorbar_output = None
            self.bali_colorbar_container = None
            self.bali_adapter = None
            self.vmin_widget = None
            self.vmax_widget = None

    def _create_bali_colorbar(self):
        """Create and display the BALI colorbar."""
        if (
            not self.show_bali
            or not self.bali_colorbar_output
            or not self.bali_adapter
        ):
            return

        segments = self.bali_adapter.get_segments_for_visualization(
            self.monitor.pid
        )
        if not segments:
            return

        # Use custom vmin/vmax if available, otherwise use data range
        if self.custom_vmin_vmax:
            vmin, vmax = self.custom_vmin_vmax
        else:
            vmin, vmax = self.bali_adapter.get_tokens_per_sec_range(segments)
            # Initialize widgets with data range on first creation
            if self.vmin_widget and self.vmax_widget:
                self.vmin_widget.value = vmin
                self.vmax_widget.value = vmax
                self.custom_vmin_vmax = (vmin, vmax)

        with self.bali_colorbar_output:
            self.bali_colorbar_output.clear_output(wait=True)
            fig = plt.figure(figsize=(8, 0.8))
            ax = fig.add_subplot(111)
            ax.set_visible(False)

            sm = ScalarMappable(
                norm=Normalize(vmin=vmin, vmax=vmax),
                cmap=self.bali_adapter.get_colormap(),
            )
            sm.set_array([])
            cbar = fig.colorbar(
                sm,
                cax=fig.add_axes([0.1, 0.3, 0.8, 0.4]),
                orientation="horizontal",
            )
            cbar.set_label("Tokens/Second", fontsize=12)
            cbar.ax.tick_params(labelsize=10)
            cbar.ax.set_facecolor("none")
            fig.patch.set_facecolor("none")
            plt.close(fig)
            display(fig)

    def display_ui(self):
        """Display the UI components."""
        ui_components = [self.add_panel_button]

        if self.show_bali:
            self._create_bali_colorbar()
            # Add observers to vmin/vmax widgets
            def on_vmin_vmax_change(change):
                if change["type"] == "change" and change["name"] == "value":
                    self.custom_vmin_vmax = (self.vmin_widget.value, self.vmax_widget.value)
                    self._create_bali_colorbar()  # Redraw colorbar
                    # Redraw all plot panels
                    for panel in self.plot_panels:
                        panel["update_plot"]()
            
            self.vmin_widget.observe(on_vmin_vmax_change)
            self.vmax_widget.observe(on_vmin_vmax_change)
            ui_components.append(self.bali_colorbar_container)

        ui_components.append(self.output_container)
        display(widgets.VBox(ui_components))
        self._on_add_panel_clicked(None)

    def _on_add_panel_clicked(self, _):
        """Add a new plot panel with dropdown and persistent matplotlib
        axis."""
        if self.panel_count >= self.max_panels:
            self.add_panel_button.disabled = True
            self.output_container.children += (
                widgets.HTML("<b>All panels have been added.</b>"),
            )
            return

        self.output_container.children += (
            widgets.HBox(
                [
                    self._create_dropdown_plot_panel(),
                    self._create_dropdown_plot_panel(),
                ],
            ),
        )
        self.panel_count += 2

        if self.panel_count >= self.max_panels:
            self.add_panel_button.disabled = True

    def _create_dropdown_plot_panel(self):
        """Create metric and level dropdown + matplotlib figure panel with
        persistent Axes."""
        metric_dropdown = widgets.Dropdown(
            options=self.labeled_options,
            value=self._get_next_metric(),
            description="Metric:",
        )
        level_dropdown = widgets.Dropdown(
            options=get_available_levels(),
            value="process",
            description="Level:",
        )
        fig, ax = plt.subplots(figsize=self.figsize, constrained_layout=True)
        if not is_ipympl_backend():
            # Prevent automatic display of the figure outside the Output widget
            plt.close(fig)
        output = widgets.Output()
        selection_output = widgets.Output()

        def update_plot():
            metric = metric_dropdown.value
            level = level_dropdown.value
            df = self.perfdata_by_level.get(level)
            # Always clear the output and redraw the figure to ensure
            # in-place updates
            output.clear_output(wait=True)
            selection_output.clear_output(wait=True)
            with output:
                ax.clear()
                # Provide a sink for BALI selection details
                ax._bali_selection_output = selection_output
                if df is not None and not df.empty:
                    self.plot_callback(
                        df,
                        metric,
                        self.cell_range,
                        self.show_idle,
                        ax,
                        level,
                        self.show_bali,
                        self.custom_vmin_vmax,
                    )
                    fig.canvas.draw_idle()
                    if not is_ipympl_backend():
                        display(fig)
                else:
                    print("No data available for the selected level/metric")

        def on_dropdown_change(change):
            if change["type"] == "change" and change["name"] == "value":
                update_plot()

        metric_dropdown.observe(on_dropdown_change)
        level_dropdown.observe(on_dropdown_change)

        # Store panel data for updates
        panel_data = {
            "metric_dropdown": metric_dropdown,
            "level_dropdown": level_dropdown,
            "figure": fig,
            "axes": ax,
            "output": output,
            "update_plot": update_plot,
        }
        self.plot_panels.append(panel_data)

        # Initial plot
        update_plot()
        if is_ipympl_backend():
            with output:
                plt.show()

        return widgets.VBox(
            [
                widgets.HBox([metric_dropdown, level_dropdown]),
                output,
                selection_output,
            ]
        )

    def _get_next_metric(self):
        for metric in self.metrics:
            if metric not in self.shown_metrics:
                self.shown_metrics.add(metric)
                return metric
        return None

    def update_data(self, perfdata_by_level, cell_range, show_idle):
        self.perfdata_by_level = perfdata_by_level
        self.cell_range = cell_range
        self.show_idle = show_idle
        for panel in self.plot_panels:
            panel["output"].clear_output(wait=True)
            panel["update_plot"]()


class DiskPerformanceVisualizer(PerformanceVisualizer):
    """Performance visualizer that loads data from disk instead of memory"""
    
    def __init__(self, pid, cell_history, bali_adapter=None):
        self.pid = pid
        self.cell_history = cell_history
        self.figsize = (5, 3)
        self.min_duration = 1.0
        self._io_window = 1
        
        # Load monitor metadata from disk
        metadata = load_monitor_metadata_from_disk(pid)
        if metadata is None:
            logger.warning(f"No monitor metadata found for PID {pid}, using defaults")
            metadata = {
                "num_cpus": 8, "num_system_cpus": 8, "num_gpus": 1,
                "gpu_memory": 30.0, "start_time": 0,
                "memory_limits": {level: 100.0 for level in get_available_levels()}
            }
        
        # Create monitor object with loaded metadata
        class MockMonitor:
            def __init__(self, pid, metadata):
                self.pid = pid
                self.num_cpus = metadata["num_cpus"]
                self.num_system_cpus = metadata["num_system_cpus"]
                self.num_gpus = metadata["num_gpus"]
                self.gpu_memory = metadata["gpu_memory"]
                self.start_time = metadata["start_time"]
                self.memory_limits = metadata["memory_limits"]
        
        self.monitor = MockMonitor(pid, metadata)
        
        # Initialize BALI functionality
        BaliVisualizationMixin.__init__(self, bali_adapter=bali_adapter)
        
        # Load perfdata from disk
        self.perfdata_by_level = load_perfdata_from_disk(pid, get_available_levels())
        
        # Initialize subsets (copy from parent class)
        self.subsets = {
            "cpu_all": {
                "cpu": {
                    "type": "multi_series",
                    "prefix": "cpu_util_",
                    "title": "CPU Utilization (%) - Across Cores",
                    "ylim": (0, 100),
                    "label": "CPU Utilization (All Cores)",
                }
            },
            "gpu_all": {
                "gpu_util": {
                    "type": "multi_series",
                    "prefix": "gpu_util_",
                    "title": "GPU Utilization (%) - Across GPUs",
                    "ylim": (0, 100),
                    "label": "GPU Utilization (All GPUs)",
                },
                "gpu_band": {
                    "type": "multi_series",
                    "prefix": "gpu_band_",
                    "title": "GPU Bandwidth Usage (%) - Across GPUs",
                    "ylim": (0, 100),
                    "label": "GPU Bandwidth (All GPUs)",
                },
                "gpu_mem": {
                    "type": "multi_series",
                    "prefix": "gpu_mem_",
                    "title": "GPU Memory Usage (GB) - Across GPUs",
                    "ylim": (0, self.monitor.gpu_memory),
                    "label": "GPU Memory (All GPUs)",
                },
                "gpu_power": {
                    "type": "multi_series",
                    "prefix": "gpu_power_",
                    "title": "GPU Power Usage (W) - Across GPUs",
                    "ylim": None,
                    "label": "GPU Power (All GPUs)",
                },
            },
            "cpu": {
                "cpu_summary": {
                    "type": "summary_series",
                    "columns": [
                        "cpu_util_min",
                        "cpu_util_avg",
                        "cpu_util_max",
                    ],
                    "title": f"CPU Utilization (%) - {self.monitor.num_cpus} CPUs",
                    "ylim": (0, 100),
                    "label": "CPU Utilization Summary",
                }
            },
            "gpu": {
                "gpu_util_summary": {
                    "type": "summary_series",
                    "columns": [
                        "gpu_util_min",
                        "gpu_util_avg",
                        "gpu_util_max",
                    ],
                    "title": f"GPU Utilization (%) - {self.monitor.num_gpus} GPUs",
                    "ylim": (0, 100),
                    "label": "GPU Utilization Summary",
                },
                "gpu_band_summary": {
                    "type": "summary_series",
                    "columns": [
                        "gpu_band_min",
                        "gpu_band_avg",
                        "gpu_band_max",
                    ],
                    "title": f"GPU Bandwidth Usage (%) - {self.monitor.num_gpus} GPUs",
                    "ylim": (0, 100),
                    "label": "GPU Bandwidth Summary",
                },
                "gpu_mem_summary": {
                    "type": "summary_series",
                    "columns": ["gpu_mem_min", "gpu_mem_avg", "gpu_mem_max"],
                    "title": f"GPU Memory Usage (GB) - {self.monitor.num_gpus} GPUs",
                    "ylim": (0, self.monitor.gpu_memory),
                    "label": "GPU Memory Summary",
                },
                "gpu_power_summary": {
                    "type": "summary_series",
                    "columns": [
                        "gpu_power_min",
                        "gpu_power_avg",
                        "gpu_power_max",
                    ],
                    "title": f"GPU Power Usage (W) - {self.monitor.num_gpus} GPUs",
                    "ylim": None,
                    "label": "GPU Power Summary",
                },
            },
            "mem": {
                "memory": {
                    "type": "single_series",
                    "column": "memory",
                    "title": "Memory Usage (GB)",
                    "ylim": None,
                    "label": "Memory Usage",
                }
            },
            "io": {
                "io_read": {
                    "type": "single_series",
                    "column": "io_read",
                    "title": "I/O Read (MB/s)",
                    "ylim": None,
                    "label": "IO Read MB/s",
                },
                "io_write": {
                    "type": "single_series",
                    "column": "io_write",
                    "title": "I/O Write (MB/s)",
                    "ylim": None,
                    "label": "IO Write MB/s",
                },
                "io_read_count": {
                    "type": "single_series",
                    "column": "io_read_count",
                    "title": "I/O Read Operations (ops/s)",
                    "ylim": None,
                    "label": "IO Read Ops",
                },
                "io_write_count": {
                    "type": "single_series",
                    "column": "io_write_count",
                    "title": "I/O Write Operations (ops/s)",
                    "ylim": None,
                    "label": "IO Write Ops",
                },
            },
        }

    def plot(self, metric_subsets=("cpu", "mem", "io"), cell_range=None, show_idle=False):
        """Plot performance metrics using disk data"""
        if any(not df.empty for df in self.perfdata_by_level.values()):
            # Override the plot method to use disk data
            self._plot_with_disk_data(metric_subsets, cell_range, show_idle)
        else:
            logger.warning("No performance data found on disk")

    def _plot_with_disk_data(self, metric_subsets, cell_range, show_idle):
        """Modified plot method that uses pre-loaded disk data"""
        # Use the parent class plot method but override data access
        original_data_view = getattr(self.monitor, 'data', None)
        
        # Create a mock data object
        class MockData:
            def view(self, level):
                return self.perfdata_by_level.get(level, pd.DataFrame())
        
        mock_data = MockData()
        mock_data.perfdata_by_level = self.perfdata_by_level
        self.monitor.data = mock_data
        
        # Call parent plot method
        super().plot(metric_subsets, cell_range, show_idle)
