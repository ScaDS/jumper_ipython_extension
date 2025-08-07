import random
from typing import List

import matplotlib.pyplot as plt
from ipywidgets import widgets
from IPython.display import display

from .utilities import filter_perfdata


class PerformanceVisualizer:
    def __init__(self, monitor, cell_history, min_duration=None):

        self.monitor = monitor
        self.cell_history = cell_history
        self.figsize = (5, 3)
        self.min_duration = min_duration

        # Metrics configuration
        self.subsets = {
            "cpu_all": {
                "cpu": {
                    "type": "multi_series",
                    "prefix": "cpu_util_",
                    "avg_column": "cpu_util_avg",
                    "title": f"CPU Utilization (%) - Across Cores",
                    "ylim": (-1, 101),
                }
            },
            "gpu_all": {
                "gpu_util": {
                    "type": "multi_series",
                    "prefix": "gpu_util_",
                    "avg_column": "gpu_util_avg",
                    "title": f"GPU Utilization (%) - Across GPUs",
                    "ylim": (-1, 101),
                },
                "gpu_band": {
                    "type": "multi_series",
                    "prefix": "gpu_band_",
                    "avg_column": "gpu_band_avg",
                    "title": f"GPU Bandwidth Usage (%) - Across GPUs",
                    "ylim": (-1, 101),
                },
                "gpu_mem": {
                    "type": "multi_series",
                    "prefix": "gpu_mem_",
                    "avg_column": "gpu_mem_avg",
                    "title": f"GPU Memory Usage (GB) - Across GPUs",
                    "ylim": (0, self.monitor.gpu_memory),
                },
            },
            "cpu": {
                "cpu_summary": {
                    "type": "summary_series",
                    "columns": ["cpu_util_min", "cpu_util_avg",
                                "cpu_util_max"],
                    "labels": ["Min", "Average", "Max"],
                    "title": f"CPU Utilization (%) - {self.monitor.num_cpus} CPUs",
                    "ylim": (-1, 101),
                }
            },
            "gpu": {
                "gpu_util_summary": {
                    "type": "summary_series",
                    "columns": ["gpu_util_min", "gpu_util_avg",
                                "gpu_util_max"],
                    "labels": ["Min", "Average", "Max"],
                    "title": f"GPU Utilization (%) - {self.monitor.num_gpus} GPUs",
                    "ylim": (-1, 101),
                },
                "gpu_band_summary": {
                    "type": "summary_series",
                    "columns": ["gpu_band_min", "gpu_band_avg",
                                "gpu_band_max"],
                    "labels": ["Min", "Average", "Max"],
                    "title": f"GPU Bandwidth Usage (%) - {self.monitor.num_gpus} GPUs",
                    "ylim": (-1, 101),
                },
                "gpu_mem_summary": {
                    "type": "summary_series",
                    "columns": ["gpu_mem_min", "gpu_mem_avg", "gpu_mem_max"],
                    "labels": ["Min", "Average", "Max"],
                    "title": f"GPU Memory Usage (GB) - {self.monitor.num_gpus} GPUs",
                    "ylim": (0, self.monitor.gpu_memory),
                },
            },
            "mem": {
                "memory": {
                    "type": "single_series",
                    "column": "memory",
                    "title": "Memory Usage (GB)",
                    "ylim": (0, self.monitor.memory),
                }
            },
            "io": {
                "io_read": {
                    "type": "single_series",
                    "column": "io_read",
                    "title": "I/O Read (MB)",
                },
                "io_write": {
                    "type": "single_series",
                    "column": "io_write",
                    "title": "I/O Write (MB)",
                },
                "io_read_count": {
                    "type": "single_series",
                    "column": "io_read_count",
                    "title": "I/O Read Operations Count",
                },
                "io_write_count": {
                    "type": "single_series",
                    "column": "io_write_count",
                    "title": "I/O Write Operations Count",
                },
            },
        }

    def _compress_time_axis(self, perfdata, cell_range):
        """Compress time axis by removing idle periods between cells"""
        if perfdata.empty:
            return perfdata, []

        start_idx, end_idx = cell_range
        cell_data = self.cell_history.view(start_idx, end_idx + 1)

        # Create compressed time mapping
        compressed_perfdata = perfdata.copy()
        cell_boundaries = []
        current_time = 0

        for idx, cell in cell_data.iterrows():
            # Find perfdata points for this cell
            cell_mask = (perfdata["time"] >= cell["start_time"]) & (
                    perfdata["time"] <= cell["end_time"]
            )
            cell_perfdata = perfdata[cell_mask]

            if not cell_perfdata.empty:
                # Calculate cell duration and compressed time mapping
                original_start = cell["start_time"]
                original_end = cell["end_time"]
                cell_duration = original_end - original_start

                # Map original times to compressed times
                original_times = cell_perfdata["time"].values
                relative_times = original_times - original_start
                compressed_times = current_time + relative_times

                # Update compressed perfdata
                compressed_perfdata.loc[cell_mask, "time"] = compressed_times

                # Store cell boundary info
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
        # Find config for this metric
        config = next(
            (subset[metric] for subset in self.subsets.values() if
             metric in subset),
            None,
        )
        if config is None:
            return

        # Check if required columns exist
        if config["type"] == "single_series" and config[
            "column"] not in df.columns:
            return
        elif config["type"] == "summary_series":
            available_cols = [col for col in config["columns"] if
                              col in df.columns]
            if not available_cols:
                return
        elif config["type"] == "multi_series":
            series_cols = [
                col
                for col in df.columns
                if col.startswith(config["prefix"]) and not col.endswith("avg")
            ]
            if config["avg_column"] not in df.columns and not series_cols:
                return

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

            # Plot based on type
        if config["type"] == "single_series":
            ax.plot(df["time"], df[config["column"]], color="blue",
                    linewidth=2)

        elif config["type"] == "summary_series":
            line_styles = ["dotted", "-", "--"]
            alpha = [0.35, 1.0, 0.35]
            for i, (col, label) in enumerate(
                    zip(config["columns"], config["labels"])):
                if col in df.columns:
                    ax.plot(
                        df["time"], df[col],
                        color="blue",
                        linestyle=line_styles[i],
                        linewidth=2,
                        alpha=alpha[i],
                        label=label
                    )
            ax.legend()

        elif config["type"] == "multi_series":
            series_cols = [
                col
                for col in df.columns
                if col.startswith(config["prefix"]) and not col.endswith("avg")
            ]
            for col in series_cols:
                ax.plot(df["time"], df[col], "-", alpha=0.5, label=col)
            if config["avg_column"] in df.columns:
                ax.plot(
                    df["time"],
                    df[config["avg_column"]],
                    "b-",
                    linewidth=2,
                    label="Mean",
                )
            ax.legend()

        # Apply settings
        title = config["title"]
        if not show_idle:
            title += " (No Idle)"
        ax.set_title(title)
        ax.set_xlabel("Time (seconds)")
        ax.grid(True)
        if "ylim" in config:
            ax.set_ylim(config["ylim"])
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
        x_max = ax.get_xlim()[1]

        min_duration = self.min_duration if self.min_duration is not None else 0

        if not show_idle and hasattr(self, "_compressed_cell_boundaries"):
            cells_list = [x for x in self._compressed_cell_boundaries if x["start_time"] < x_max
                          and x["end_time"] > 0 and x["duration"] > min_duration]
            # Use compressed boundaries for no_idle mode
            for cell in cells_list:
                duration = cell["duration"]
                start_time = cell["start_time"]
                cell_num = int(cell["index"])

                if len(cells_list) == 1:
                    # avoid coloring if there is only one cell
                    color = 'none'
                else:
                    color = colors[cell_num]

                width = duration
                height = y_max - y_min
                # Add rectangle and label
                ax.add_patch(
                    plt.Rectangle(
                        (cell["start_time"], y_min),
                        width,
                        height,
                        facecolor=color,
                        alpha=0.4,
                        edgecolor="black",
                        linestyle="--",
                        linewidth=1,
                        zorder=0,
                    )
                )
                ax.text(
                    start_time + width / 2,
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
        else:
            # Original implementation for normal mode
            filtered_cells = self.cell_history.view()
            if cell_range:
                start_idx, end_idx = cell_range
                cells = filtered_cells.iloc[start_idx:end_idx + 1]
            else:
                cells = filtered_cells

            for idx, cell in cells.iterrows():
                # Calculate adjusted times
                start_time = cell["start_time"] - self.monitor.start_time
                end_time = cell["end_time"] - self.monitor.start_time
                duration = cell["duration"]
                # Skip cells outside visible range or too short
                if end_time < 0 or start_time > x_max or duration < min_duration:
                    continue
                cell_num = int(cell["index"])
                color = colors[cell_num]
                if cells.shape[1] == 1:
                    # avoid coloring if there is only one cell
                    color = 'none'
                width = duration
                height = y_max - y_min
                # Add rectangle and label
                ax.add_patch(
                    plt.Rectangle(
                        (start_time, y_min),
                        width,
                        height,
                        facecolor=color,
                        alpha=0.5,
                        edgecolor="black",
                        linestyle="--",
                        linewidth=1,
                        zorder=0,
                    )
                )
                ax.text(
                    start_time + width / 2,
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

    def plot(
            self,
            metric_subsets=("cpu", "cpu_all", "mem", "io"),
            cell_range=None,
            show_idle=False,
    ):

        # Determine cell indices for slicing
        if cell_range is None:
            valid_cells = self.cell_history.view()
            if len(valid_cells) > 0:
                last_valid_cell_idx = int(valid_cells.iloc[-1]["index"])
                cell_range = (last_valid_cell_idx, last_valid_cell_idx)
            else:
                return

        start_idx, end_idx = cell_range
        filtered_cells = self.cell_history.view(start_idx, end_idx + 1)

        perfdata = self.monitor.data.view()
        perfdata = filter_perfdata(
            filtered_cells, perfdata, not show_idle
        )

        if perfdata.empty:
            print("No performance data available")
            return

        # Handle time compression (default) or show idle if requested
        if not show_idle:
            perfdata, self._compressed_cell_boundaries = self._compress_time_axis(
                perfdata, cell_range
            )
        else:
            perfdata = perfdata.copy()
            perfdata["time"] -= self.monitor.start_time

        # Get metric names for given subsets
        metrics = []
        for subset in metric_subsets:
            if subset in self.subsets:
                metrics.extend(self.subsets[subset].keys())
            else:
                print(f"Unknown metric subset: {subset}")

        plot_wrapper = InteractivePlotWrapper(
            plot_callback=self._plot_metric,
            metrics=metrics,
            df=perfdata,
            cell_range=cell_range,
            show_idle=show_idle,
            figsize=self.figsize,
        )
        plot_wrapper.display_ui()


class InteractivePlotWrapper:
    """
    Interactive plotter:
    - Each dropdown allows independent selection from all available metrics.
    - Matplotlib axes are reused with ax.clear() instead of clearing output.
    """

    def __init__(
            self,
            plot_callback,
            df,
            metrics: List[str],
            cell_range=None,
            show_idle=False,
            figsize: tuple = None
    ):
        """
        Parameters:
            plot_callback: function(df, metric_name, show_cells, cell_history, ax)
            metrics: list of metric names (str)
            df: pandas.DataFrame
            cell_range:
            show_idle:
        """
        self.plot_callback = plot_callback
        self.df = df
        self.metrics = metrics
        self.shown_metrics = set()
        self.cell_range = cell_range
        self.show_idle = show_idle
        self.figsize = figsize

        self.panel_count = 0
        self.max_panels = len(metrics)

        self.output_container = widgets.VBox()
        self.add_panel_button = widgets.Button(description="Add Plot Panel")
        self.add_panel_button.on_click(self._on_add_panel_clicked)

    def display_ui(self):
        """
        Displays the Add button and all interactive panels.        """
        display(widgets.VBox([self.add_panel_button, self.output_container]))

        # Create an initial panel
        self._on_add_panel_clicked(None)

    def _on_add_panel_clicked(self, _):
        """
        Adds a new plot panel with dropdown and a persistent matplotlib axis.        """
        if self.panel_count >= self.max_panels:
            self.add_panel_button.disabled = True
            self.output_container.children += (
            widgets.HTML("<b>All panels have been added.</b>"),)
            return

        panel = self._create_double_plot_panel()
        self.output_container.children += (panel,)
        self.panel_count += 2

        if self.panel_count >= self.max_panels:
            self.add_panel_button.disabled = True

    def _create_double_plot_panel(self):
        return widgets.HBox([
            self._create_dropdown_plot_panel(),
            self._create_dropdown_plot_panel(),
        ])

    def _create_dropdown_plot_panel(self):
        """
        Creates one dropdown + matplotlib figure panel with persistent Axes.        """
        dropdown = widgets.Dropdown(
            options=self.metrics,
            value=self._get_next_metric(),
            description="Metric:",
        )

        fig, ax = plt.subplots(figsize=self.figsize)
        output = widgets.Output()

        def on_dropdown_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                with output:
                    ax.clear()
                    self.plot_callback(self.df, change['new'], self.cell_range,
                                       self.show_idle, ax)
                    fig.canvas.draw_idle()

        dropdown.observe(on_dropdown_change)

        # Initial plot
        with output:
            self.plot_callback(self.df, dropdown.value, self.cell_range,
                               self.show_idle, ax)
            plt.show()

        return widgets.VBox([dropdown, output])

    def _get_next_metric(self):
        for metric in self.metrics:
            if metric not in self.shown_metrics:
                self.shown_metrics.add(metric)
                return metric
        return None
