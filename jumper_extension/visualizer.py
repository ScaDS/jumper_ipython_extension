import random
from typing import List

import matplotlib.pyplot as plt
from IPython.display import display
from ipywidgets import widgets

from .utilities import filter_perfdata


class PerformanceVisualizer:
    """Visualizes performance metrics collected by PerformanceMonitor.

    Supports four levels: 'user', 'process' (default), 'system', 'slurm'
    """

    def __init__(self, monitor, cell_history, min_duration=None):

        self.monitor = monitor
        self.cell_history = cell_history
        self.figsize = (5, 3)
        self.min_duration = min_duration

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
                    (0, monitor.memory),
                )
            },
            "io": {
                "io_read": ("single_series", "io_read", "I/O Read (MB)", None),
                "io_write": (
                "single_series", "io_write", "I/O Write (MB)", None),
                "io_read_count": (
                    "single_series",
                    "io_read_count",
                    "I/O Read Operations Count",
                    None,
                ),
                "io_write_count": (
                    "single_series",
                    "io_write_count",
                    "I/O Write Operations Count",
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
            self, df, metric, cell_range=None, show_idle=False,
            ax: plt.Axes = None
    ):
        """Plot a single metric using its configuration"""
        config = next(
            (subset[metric] for subset in self.subsets.values() if
             metric in subset),
            None,
        )
        if not config:
            return

        # Parse compressed config format
        if len(config) == 4:  # single_series: (type, column, title, ylim)
            plot_type, column, title, ylim = config
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
            available_cols = [col for col in columns if col in df.columns]
            if not available_cols:
                return
        else:
            return

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        # Plot based on type
        if plot_type == "single_series":
            ax.plot(df["time"], df[column], color="blue", linewidth=2)
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
                ax.plot(df["time"], df[avg_column], "b-", linewidth=2,
                        label="Mean")
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
            "#"
            + "".join([random.choice("0123456789ABCDEF") for _ in range(6)])
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
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.8),
            )

        if not show_idle and hasattr(self, "_compressed_cell_boundaries"):
            for cell in self._compressed_cell_boundaries:
                draw_cell_rect(
                    cell["start_time"], cell["duration"], int(cell["index"]),
                    0.4
                )
        else:
            filtered_cells = self.cell_history.view()
            cells = (
                filtered_cells.iloc[cell_range[0]: cell_range[1] + 1]
                if cell_range
                else filtered_cells
            )
            for idx, cell in cells.iterrows():
                start_time = cell["start_time"] - self.monitor.start_time
                draw_cell_rect(start_time, cell["duration"],
                               int(cell["index"]), 0.5)

    def plot(
            self,
            metric_subsets=("cpu", "cpu_all", "mem", "io"),
            cell_range=None,
            show_idle=False,
            level="process",
    ):
        if self.monitor.num_gpus:
            metric_subsets += ("gpu", "gpu_all",)

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
            value=show_idle, description="Show idle periods", style=style
        )
        cell_range_slider = widgets.IntRangeSlider(
            value=cell_range,
            min=min_cell_idx,
            max=max_cell_idx,
            step=1,
            description="Cell range:",
            style=style,
        )
        level_dropdown = widgets.Dropdown(
            options=["user", "process", "system", "slurm"],
            value=level,
            description="Performance level:",
            style=style,
        )

        config_widgets = widgets.VBox(
            [
                widgets.HTML("<b>Plot Configuration:</b>"),
                show_idle_checkbox,
                cell_range_slider,
                level_dropdown,
            ]
        )
        plot_output = widgets.Output()

        def update_plots():
            current_cell_range, current_show_idle, current_level = (
                cell_range_slider.value,
                show_idle_checkbox.value,
                level_dropdown.value,
            )
            start_idx, end_idx = current_cell_range
            filtered_cells = self.cell_history.view(start_idx, end_idx + 1)
            perfdata = filter_perfdata(
                filtered_cells,
                self.monitor.data.view(level=current_level),
                not current_show_idle,
            )

            if perfdata.empty:
                with plot_output:
                    plot_output.clear_output()
                    print("No performance data available for selected range")
                return

            # Handle time compression or show idle
            if not current_show_idle:
                perfdata, self._compressed_cell_boundaries = self._compress_time_axis(
                    perfdata, current_cell_range
                )
            else:
                perfdata = perfdata.copy()
                perfdata["time"] -= self.monitor.start_time

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
                    perfdata,
                    current_cell_range,
                    current_show_idle,
                    self.figsize,
                ).display_ui()

        # Set up observers and display
        for widget in [show_idle_checkbox, cell_range_slider, level_dropdown]:
            widget.observe(lambda change: update_plots(), names="value")

        display(widgets.VBox([config_widgets, plot_output]))
        update_plots()


class InteractivePlotWrapper:
    """Interactive plotter with dropdown selection and reusable matplotlib axes."""

    def __init__(
            self,
            plot_callback,
            metrics: List[str],
            df,
            cell_range=None,
            show_idle=False,
            figsize=None,
    ):
        self.plot_callback, self.df, self.metrics = plot_callback, df, metrics
        self.cell_range, self.show_idle, self.figsize = cell_range, show_idle, figsize
        self.shown_metrics, self.panel_count, self.max_panels = set(), 0, len(
            metrics)
        self.output_container = widgets.VBox()
        self.add_panel_button = widgets.Button(description="Add Plot Panel")
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
                [self._create_dropdown_plot_panel(),
                 self._create_dropdown_plot_panel()]
            ),
        )
        self.panel_count += 2

        if self.panel_count >= self.max_panels:
            self.add_panel_button.disabled = True

    def _create_dropdown_plot_panel(self):
        """Create one dropdown + matplotlib figure panel with persistent Axes."""
        dropdown = widgets.Dropdown(
            options=self.metrics, value=self._get_next_metric(),
            description="Metric:"
        )
        fig, ax = plt.subplots(figsize=self.figsize)
        output = widgets.Output()

        def on_dropdown_change(change):
            if change["type"] == "change" and change["name"] == "value":
                with output:
                    ax.clear()
                    self.plot_callback(
                        self.df, change["new"], self.cell_range,
                        self.show_idle, ax
                    )
                    fig.canvas.draw_idle()

        dropdown.observe(on_dropdown_change)

        # Initial plot
        with output:
            self.plot_callback(
                self.df, dropdown.value, self.cell_range, self.show_idle, ax
            )
            plt.show()

        return widgets.VBox([dropdown, output])

    def _get_next_metric(self):
        for metric in self.metrics:
            if metric not in self.shown_metrics:
                self.shown_metrics.add(metric)
                return metric
        return None
