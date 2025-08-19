import re
from typing import List

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from IPython.display import display
from ipywidgets import widgets, Layout

from .utilities import filter_perfdata, get_available_levels
from .logo import logo_image, jumper_colors
from .bali_hook import BaliResultsParser


class PerformanceVisualizer:
    """Visualizes performance metrics collected by PerformanceMonitor.

    Supports multiple levels: 'user', 'process' (default), 'system', and
    'slurm' (if available)
    """

    def __init__(self, monitor, cell_history, min_duration=None):
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

        # Initialize BALI hook
        self.bali_parser = BaliResultsParser()

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
                "io_read": (
                    "single_series",
                    "io_read",
                    "I/O Read (MB/s)",
                    None,
                ),
                "io_write": (
                    "single_series",
                    "io_write",
                    "I/O Write (MB/s)",
                    None,
                ),
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

    def _compress_bali_segments(self, bali_segments, cell_range, perfdata):
        """Compress BALI segments to match compressed time axis exactly
        (mirrors _compress_time_axis)."""
        if not bali_segments:
            return []
        start_idx, end_idx = cell_range
        cell_data = self.cell_history.view(start_idx, end_idx + 1)
        compressed, current_time, processed = [], 0, set()
        for _, cell in cell_data.iterrows():
            cell_mask = (perfdata["time"] >= cell["start_time"]) & (
                perfdata["time"] <= cell["end_time"]
            )
            if perfdata[cell_mask].empty:
                continue
            cell_start, cell_end = cell["start_time"], cell["end_time"]
            cell_duration = cell_end - cell_start
            for i, seg in enumerate(bali_segments):
                seg_start, seg_end = seg["start_time"], seg["end_time"]
                seg_id = (i, cell["index"])
                if seg_id in processed or not (
                    seg_start < cell_end and seg_end > cell_start
                ):
                    continue
                overlap_start, overlap_end = max(seg_start, cell_start), min(
                    seg_end, cell_end
                )
                if overlap_start < overlap_end:
                    start = current_time + (overlap_start - cell_start)
                    dur = overlap_end - overlap_start
                    compressed.append(
                        {
                            "start_time": start,
                            "end_time": start + dur,
                            "duration": dur,
                            "tokens_per_sec": seg.get("tokens_per_sec"),
                            "framework": seg.get("framework"),
                            "iteration": seg.get("iteration"),
                            # pass-through optional metadata for hover
                            "model": seg.get("model"),
                            "batch_size": seg.get("batch_size"),
                            "input_len": seg.get("input_len"),
                            "output_len": seg.get("output_len"),
                        }
                    )
                    processed.add(seg_id)
            current_time += cell_duration
        return compressed

    def _plot_metric(
        self,
        df,
        metric,
        cell_range=None,
        show_idle=False,
        ax: plt.Axes = None,
        level="process",
        show_bali=False,
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
        self._draw_bali_segments(ax, show_bali, show_idle, cell_range)

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
        self, ax, show_bali=False, show_idle=True, cell_range=None
    ):
        """Draw BALI benchmark segments as colored rectangles with
        tokens/sec colormap and enable hover tooltips for details."""
        if not show_bali:
            return
        try:
            segments = self.bali_parser.collect_all_bali_segments(
                self.monitor.pid
            )
            if not segments:
                return
            y_min, y_max = ax.get_ylim()
            x_max, height = ax.get_xlim()[1], y_max - y_min
            if not show_idle and hasattr(self, "_compressed_bali_segments"):
                draw_segments = self._compressed_bali_segments
                vmin, vmax = self.bali_parser.get_tokens_per_sec_range(
                    draw_segments
                )
            else:
                draw_segments = [
                    {
                        **s,
                        "start_time": s["start_time"]
                        - self.monitor.start_time,
                    }
                    for s in segments
                ]
                vmin, vmax = self.bali_parser.get_tokens_per_sec_range(
                    segments
                )
            # Reset hover handler if present from previous draw
            if hasattr(ax, "_bali_hover") and isinstance(ax._bali_hover, dict):
                try:
                    ax.figure.canvas.mpl_disconnect(ax._bali_hover.get("cid"))
                except Exception:
                    pass
                ax._bali_hover = None
            ax._bali_patches = []
            # Prepare hover annotation
            annot = ax.annotate(
                "",
                xy=(0, 0),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="red", alpha=0.9),
                fontsize=8,
            )
            annot.set_visible(False)
            for s in draw_segments:
                start, dur, tps = (
                    s["start_time"],
                    s["duration"],
                    s["tokens_per_sec"],
                )
                if start > x_max or start + dur < 0:
                    continue
                color = self.bali_parser.get_color_for_tokens_per_sec(
                    tps, vmin, vmax
                )
                rect = plt.Rectangle(
                    (start, y_min),
                    dur,
                    height,
                    facecolor=color,
                    alpha=0.75,
                    edgecolor="red",
                    linestyle="-",
                    linewidth=0.5,
                    zorder=0.5,
                )
                # store metadata for hover
                rect._bali_info = {
                    "model": s.get("model"),
                    "framework": s.get("framework"),
                    "batch_size": s.get("batch_size"),
                    "input_len": s.get("input_len"),
                    "output_len": s.get("output_len"),
                }
                ax.add_patch(rect)
                ax._bali_patches.append(rect)

            # Hover callback
            def _format_value(v):
                return "n/a" if v is None or v == "" else str(v)

            def on_move(event):
                if event.inaxes != ax:
                    if annot.get_visible():
                        annot.set_visible(False)
                        ax.figure.canvas.draw_idle()
                    return
                # Find first patch under cursor
                for patch in ax._bali_patches:
                    contains, _ = patch.contains(event)
                    if contains:
                        info = getattr(patch, "_bali_info", {})
                        text = (
                            f"Model: {_format_value(info.get('model'))}\n"
                            f"Framework: {_format_value(info.get('framework'))}\n"
                            f"Batch size: {_format_value(info.get('batch_size'))}\n"
                            f"Input len: {_format_value(info.get('input_len'))}\n"
                            f"Output len: {_format_value(info.get('output_len'))}"
                        )
                        annot.xy = (event.xdata, event.ydata)
                        annot.set_text(text)
                        annot.set_visible(True)
                        ax.figure.canvas.draw_idle()
                        return
                if annot.get_visible():
                    annot.set_visible(False)
                    ax.figure.canvas.draw_idle()

            cid = ax.figure.canvas.mpl_connect("motion_notify_event", on_move)
            ax._bali_hover = {"annot": annot, "cid": cid}
        except Exception:
            pass

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
            print("No cell history available")
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
                    print("No performance data available for selected range")
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
            if current_show_bali:
                bali_segments = self.bali_parser.collect_all_bali_segments(
                    self.monitor.pid
                )
                if not current_show_idle:
                    primary_level = get_available_levels()[0]
                    reference_perfdata = perfdata_by_level.get(primary_level)
                    self._compressed_bali_segments = (
                        self._compress_bali_segments(
                            bali_segments,
                            current_cell_range,
                            reference_perfdata,
                        )
                        if (
                            reference_perfdata is not None
                            and not reference_perfdata.empty
                        )
                        else []
                    )
                else:
                    self._compressed_bali_segments = None

            # Get metrics for subsets
            metrics = []
            for subset in metric_subsets:
                if subset in self.subsets:
                    metrics.extend(self.subsets[subset].keys())
                else:
                    print(f"Unknown metric subset: {subset}")

            with plot_output:
                # Reuse existing wrapper when possible for smoother updates.
                # Recreate only if it doesn't exist or if BALI toggle state
                # changed.
                if (
                    plot_wrapper is None
                    or getattr(plot_wrapper, "show_bali", None)
                    != current_show_bali
                ):
                    plot_output.clear_output()
                    plot_wrapper = InteractivePlotWrapper(
                        self._plot_metric,
                        metrics,
                        processed_perfdata,
                        current_cell_range,
                        current_show_idle,
                        current_show_bali,
                        self.figsize,
                    )
                    # Provide monitor reference for BALI PID access and
                    # colorbar range
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
        self.bali_colorbar_output = widgets.Output() if show_bali else None
        # Center the colorbar horizontally within the UI
        self.bali_colorbar_container = (
            widgets.HBox(
                [self.bali_colorbar_output],
                layout=Layout(
                    display="flex", justify_content="center", width="100%"
                ),
            )
            if show_bali
            else None
        )
        self.bali_parser = BaliResultsParser()

    def _create_bali_colorbar(self):
        """Create and display the global BALI colorbar showing tokens/sec
        range."""
        if not self.show_bali or not self.bali_colorbar_output:
            return
        try:
            pid = getattr(getattr(self, "monitor", None), "pid", 0)
            segments = self.bali_parser.collect_all_bali_segments(pid)
            if not segments:
                return
            vmin, vmax = self.bali_parser.get_tokens_per_sec_range(segments)
            with self.bali_colorbar_output:
                self.bali_colorbar_output.clear_output(wait=True)
                fig = plt.figure(figsize=(8, 0.8))
                gs = fig.add_gridspec(1, 1, figure=fig, hspace=0, wspace=0)
                ax = fig.add_subplot(gs[0, 0])
                ax.set_visible(False)
                sm = ScalarMappable(
                    norm=Normalize(vmin=vmin, vmax=vmax),
                    cmap=self.bali_parser.colormap,
                )
                sm.set_array([])
                cbar = fig.colorbar(
                    sm,
                    cax=fig.add_axes([0.1, 0.3, 0.8, 0.4]),
                    orientation="horizontal",
                )
                cbar.set_label("Tokens/Second", fontsize=12, fontweight="bold")
                cbar.ax.tick_params(labelsize=10)
                plt.close(fig)
                display(fig)
        except Exception:
            pass

    def display_ui(self):
        """Display the Add button, BALI colorbar (if enabled), and all
        interactive panels."""
        ui_components = [self.add_panel_button]

        if self.show_bali and self.bali_colorbar_output:
            self._create_bali_colorbar()
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
            options=self.metrics,
            value=self._get_next_metric(),
            description="Metric:",
        )
        level_dropdown = widgets.Dropdown(
            options=get_available_levels(),
            value="process",
            description="Level:",
        )
        fig, ax = plt.subplots(figsize=self.figsize, constrained_layout=True)
        # Prevent automatic display of the figure outside the Output widget
        plt.close(fig)
        output = widgets.Output()

        def update_plot():
            metric = metric_dropdown.value
            level = level_dropdown.value
            df = self.perfdata_by_level.get(level)
            # Always clear the output and redraw the figure to ensure
            # in-place updates
            output.clear_output(wait=True)
            with output:
                ax.clear()
                if df is not None and not df.empty:
                    self.plot_callback(
                        df,
                        metric,
                        self.cell_range,
                        self.show_idle,
                        ax,
                        level,
                        self.show_bali,
                    )
                    fig.canvas.draw_idle()
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

        return widgets.VBox(
            [widgets.HBox([metric_dropdown, level_dropdown]), output]
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
