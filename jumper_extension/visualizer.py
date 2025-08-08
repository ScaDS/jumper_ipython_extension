import random
import re
from typing import List

import matplotlib.pyplot as plt
from IPython.display import display
from ipywidgets import widgets, Layout

from .utilities import filter_perfdata, get_available_levels
from .logo import logo_image


class PerformanceVisualizer:
    """Visualizes performance metrics collected by PerformanceMonitor.

    Supports multiple levels: 'user', 'process' (default), 'system', and 'slurm' (if available)
    """

    def __init__(self, monitor, cell_history, min_duration=None):
        self.monitor = monitor
        self.cell_history = cell_history
        self.figsize = (5, 3)
        self.min_duration = min_duration
        # Smooth IO with ~1s rolling window based on sampling interval
        try:
            self._io_window = max(1, int(round(1.0 / (self.monitor.interval or 1.0))))
        except Exception:
            self._io_window = 1

        # Compressed metrics configuration
        self.subsets = {
            "cpu_all": {
                "cpu": (
                    "multi_series",
                    "cpu_util_",
                    "cpu_util_avg",
                    "CPU Utilization (%) - Across Cores",
                    (0, 100),
                )
            },
            "gpu_all": {
                "gpu_util": (
                    "multi_series",
                    "gpu_util_",
                    "gpu_util_avg",
                    "GPU Utilization (%) - Across GPUs",
                    (0, 100),
                ),
                "gpu_band": (
                    "multi_series",
                    "gpu_band_",
                    "gpu_band_avg",
                    "GPU Bandwidth Usage (%) - Across GPUs",
                    (0, 100),
                ),
                "gpu_mem": (
                    "multi_series",
                    "gpu_mem_",
                    "gpu_mem_avg",
                    "GPU Memory Usage (GB) - Across GPUs",
                    (0, monitor.gpu_memory),
                ),
            },
            "cpu": {
                "cpu_summary": (
                    "summary_series",
                    ["cpu_util_min", "cpu_util_avg", "cpu_util_max"],
                    ["Min", "Average", "Max"],
                    f"CPU Utilization (%) - {self.monitor.num_cpus} CPUs",
                    (0, 100),
                )
            },
            "gpu": {
                "gpu_util_summary": (
                    "summary_series",
                    ["gpu_util_min", "gpu_util_avg", "gpu_util_max"],
                    ["Min", "Average", "Max"],
                    f"GPU Utilization (%) - {self.monitor.num_gpus} GPUs",
                    (0, 100),
                ),
                "gpu_band_summary": (
                    "summary_series",
                    ["gpu_band_min", "gpu_band_avg", "gpu_band_max"],
                    ["Min", "Average", "Max"],
                    f"GPU Bandwidth Usage (%) - {self.monitor.num_gpus} GPUs",
                    (0, 100),
                ),
                "gpu_mem_summary": (
                    "summary_series",
                    ["gpu_mem_min", "gpu_mem_avg", "gpu_mem_max"],
                    ["Min", "Average", "Max"],
                    f"GPU Memory Usage (GB) - {self.monitor.num_gpus} GPUs",
                    (0, monitor.gpu_memory),
                ),
            },
            "mem": {
                "memory": (
                    "single_series",
                    "memory",
                    "Memory Usage (GB)",
                    None,  # Will be set dynamically based on level
                )
            },
            "io": {
                "io_read": ("single_series", "io_read", "I/O Read (MB/s)", None),
                "io_write": ("single_series", "io_write", "I/O Write (MB/s)", None),
                "io_read_count": (
                    "single_series",
                    "io_read_count",
                    "I/O Read Operations (ops/s)",
                    None,
                ),
                "io_write_count": (
                    "single_series",
                    "io_write_count",
                    "I/O Write Operations (ops/s)",
                    None,
                ),
            },
        }

    def _compress_time_axis(self, perfdata, cell_range):
        """Compress time axis by removing idle periods between cells"""
        if perfdata.empty:
            return perfdata, []

        start_idx, end_idx = cell_range
        cell_data = self.cell_history.view(start_idx, end_idx + 1)
        compressed_perfdata, cell_boundaries, current_time = perfdata.copy(), [], 0

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
    ):
        """Plot a single metric using its configuration"""
        config = next(
            (subset[metric] for subset in self.subsets.values() if metric in subset),
            None,
        )
        if not config:
            return

        # Parse compressed config format
        if len(config) == 4:  # single_series: (type, column, title, ylim)
            plot_type, column, title, ylim = config
            # Set dynamic memory limit for memory metric
            if metric == "memory" and ylim is None:
                ylim = (0, self.monitor.memory_limits[level])
            if column not in df.columns:
                return
        elif (
            len(config) == 5 and config[0] == "multi_series"
        ):  # multi_series: (type, prefix, avg_column, title, ylim)
            plot_type, prefix, avg_column, title, ylim = config
            series_cols = [
                col
                for col in df.columns
                if col.startswith(prefix) and not col.endswith("avg")
            ]
            if avg_column not in df.columns and not series_cols:
                return
        elif (
            len(config) == 5 and config[0] == "summary_series"
        ):  # summary_series: (type, columns, labels, title, ylim)
            plot_type, columns, labels, title, ylim = config
            if level == "system":
                title = re.sub(r'\d+', str(self.monitor.num_system_cpus), title)
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
            if metric in ("io_read", "io_write", "io_read_count", "io_write_count"):
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
            for i, (col, label) in enumerate(zip(columns, labels)):
                if col in df.columns:
                    ax.plot(
                        df["time"],
                        df[col],
                        color="blue",
                        linestyle=line_styles[i],
                        linewidth=2,
                        alpha=alpha_vals[i],
                        label=label,
                    )
            ax.legend()
        elif plot_type == "multi_series":
            for col in series_cols:
                ax.plot(df["time"], df[col], "-", alpha=0.5, label=col)
            if avg_column in df.columns:
                ax.plot(df["time"], df[avg_column], "b-", linewidth=2, label="Mean")
            ax.legend()

        # Apply settings
        ax.set_title(title + (" (No Idle)" if not show_idle else ""))
        ax.set_xlabel("Time (seconds)")
        ax.grid(True)
        if ylim:
            ax.set_ylim(ylim)
        self._draw_cell_boundaries(ax, cell_range, show_idle)

    def _draw_cell_boundaries(self, ax, cell_range=None, show_idle=False):
        """Draw cell boundaries as colored rectangles with cell indices"""
        # define the seed for random color picking, i.e. to keep cells in the
        # same color when plotting in different graphs
        random.seed(1337)

        colors = [
            "#" + "".join([random.choice("0123456789ABCDEF") for _ in range(6)])
            for _ in range(len(self.cell_history))
        ]
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
                    edgecolor="black",
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
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        if not show_idle and hasattr(self, "_compressed_cell_boundaries"):
            for cell in self._compressed_cell_boundaries:
                draw_cell_rect(
                    cell["start_time"], cell["duration"], int(cell["index"]), 0.4
                )
        else:
            filtered_cells = self.cell_history.view()
            cells = (
                filtered_cells.iloc[cell_range[0]:cell_range[1] + 1]
                if cell_range
                else filtered_cells
            )
            for idx, cell in cells.iterrows():
                start_time = cell["start_time"] - self.monitor.start_time
                draw_cell_rect(start_time, cell["duration"], int(cell["index"]), 0.5)

    def plot(
        self,
        metric_subsets=("cpu", "cpu_all", "mem", "io"),
        cell_range=None,
        show_idle=False,
    ):
        if self.monitor.num_gpus:
            metric_subsets += (
                "gpu",
                "gpu_all",
            )

        """Plot performance metrics with interactive widgets for configuration."""
        valid_cells = self.cell_history.view()
        if len(valid_cells) == 0:
            print("No cell history available")
            return

        # Default to all cells if no range specified
        min_cell_idx, max_cell_idx = int(valid_cells.iloc[0]["index"]), int(
            valid_cells.iloc[-1]["index"]
        )
        if cell_range is None:
            cell_range = (min_cell_idx, max_cell_idx)

        # Create interactive widgets
        style = {"description_width": "initial"}
        show_idle_checkbox = widgets.Checkbox(
            value=show_idle, description="Show idle periods"
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
                cell_range_slider,
                logo_widget,
            ],
            layout=box_layout,
        )
        plot_output = widgets.Output()

        def update_plots():
            current_cell_range, current_show_idle = (
                cell_range_slider.value,
                show_idle_checkbox.value,
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
                    print("No performance data available for selected range")
                return

            # Handle time compression or show idle for all levels
            processed_perfdata = {}
            for level_key, perfdata in perfdata_by_level.items():
                if not perfdata.empty:
                    if not current_show_idle:
                        processed_data, self._compressed_cell_boundaries = (
                            self._compress_time_axis(perfdata, current_cell_range)
                        )
                        processed_perfdata[level_key] = processed_data
                    else:
                        processed_data = perfdata.copy()
                        processed_data["time"] -= self.monitor.start_time
                        processed_perfdata[level_key] = processed_data
                else:
                    processed_perfdata[level_key] = perfdata

            # Get metrics for subsets
            metrics = []
            for subset in metric_subsets:
                if subset in self.subsets:
                    metrics.extend(self.subsets[subset].keys())
                else:
                    print(f"Unknown metric subset: {subset}")

            with plot_output:
                plot_output.clear_output()
                InteractivePlotWrapper(
                    self._plot_metric,
                    metrics,
                    processed_perfdata,
                    current_cell_range,
                    current_show_idle,
                    self.figsize,
                ).display_ui()

        # Set up observers and display
        for widget in [show_idle_checkbox, cell_range_slider]:
            widget.observe(lambda change: update_plots(), names="value")

        display(widgets.VBox([config_widgets, plot_output]))
        update_plots()


class InteractivePlotWrapper:
    """Interactive plotter with dropdown selection and reusable matplotlib axes."""

    def __init__(
        self,
        plot_callback,
        metrics: List[str],
        perfdata_by_level,
        cell_range=None,
        show_idle=False,
        figsize=None,
    ):
        self.plot_callback, self.perfdata_by_level, self.metrics = (
            plot_callback,
            perfdata_by_level,
            metrics,
        )
        self.cell_range, self.show_idle, self.figsize = cell_range, show_idle, figsize
        self.shown_metrics, self.panel_count, self.max_panels = (
            set(),
            0,
            len(metrics) * 4,
        )
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
            description="Add Plot Panel", layout=Layout(margin="auto")
        )
        self.add_panel_button.on_click(self._on_add_panel_clicked)

    def display_ui(self):
        """Display the Add button and all interactive panels."""
        display(widgets.VBox([self.add_panel_button, self.output_container]))
        self._on_add_panel_clicked(None)

    def _on_add_panel_clicked(self, _):
        """Add a new plot panel with dropdown and persistent matplotlib axis."""
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
        """Create metric and level dropdown + matplotlib figure panel with persistent Axes."""
        metric_dropdown = widgets.Dropdown(
            options=self.metrics, value=self._get_next_metric(), description="Metric:"
        )
        level_dropdown = widgets.Dropdown(
            options=get_available_levels(), value="process", description="Level:"
        )
        fig, ax = plt.subplots(figsize=self.figsize)
        output = widgets.Output()

        def update_plot():
            metric = metric_dropdown.value
            level = level_dropdown.value
            df = self.perfdata_by_level.get(level)
            if df is not None and not df.empty:
                with output:
                    ax.clear()
                    self.plot_callback(
                        df, metric, self.cell_range, self.show_idle, ax, level
                    )
                    fig.canvas.draw_idle()

        def on_dropdown_change(change):
            if change["type"] == "change" and change["name"] == "value":
                update_plot()

        metric_dropdown.observe(on_dropdown_change)
        level_dropdown.observe(on_dropdown_change)

        # Initial plot
        update_plot()
        with output:
            plt.show()

        return widgets.VBox([widgets.HBox([metric_dropdown, level_dropdown]), output])

    def _get_next_metric(self):
        for metric in self.metrics:
            if metric not in self.shown_metrics:
                self.shown_metrics.add(metric)
                return metric
        return None
