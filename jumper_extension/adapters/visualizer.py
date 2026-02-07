import logging
import pickle
import re
from typing import List, runtime_checkable, Protocol, Optional, Tuple

import matplotlib.pyplot as plt
from IPython.display import display
from ipywidgets import widgets, Layout

from jumper_extension.adapters.cell_history import CellHistory
from jumper_extension.monitor.common import UnavailablePerformanceMonitor, \
    MonitorProtocol
from jumper_extension.core.messages import (
    ExtensionErrorCode,
    EXTENSION_ERROR_MESSAGES, ExtensionInfoCode, EXTENSION_INFO_MESSAGES,
)
from jumper_extension.utilities import filter_perfdata, get_available_levels
from jumper_extension.logo import logo_image, jumper_colors

logger = logging.getLogger("extension")


def is_ipympl_backend():
    try:
        backend = plt.get_backend().lower()
    except Exception:
        return False
    return ("ipympl" in backend) or ("widget" in backend)


@runtime_checkable
class VisualizerProtocol(Protocol):
    """Structural protocol for visualizers used by the service."""
    def attach(self, monitor: MonitorProtocol) -> None: ...
    def plot(
        self,
        metric_subsets=("cpu", "mem", "io"),
        cell_range=None,
        show_idle=False,
        level=None,
        save_jpeg=None,
        pickle_file=None,
    ) -> None: ...


class PerformanceVisualizer:
    """Visualizes performance metrics collected by PerformanceMonitor.

    Supports multiple levels: 'user', 'process' (default), 'system', and
    'slurm' (if available)
    """

    def __init__(self, cell_history: CellHistory):
        self.monitor = UnavailablePerformanceMonitor(
            reason="Monitor has not been started yet."
        )
        self.cell_history = cell_history
        self.figsize = (5, 3)
        self.min_duration = None
        self._io_window = None
        self.subsets = {}

    def attach(
        self,
        monitor: MonitorProtocol,
    ):
        """Attach started PerformanceMonitor."""
        self.monitor = monitor
        self.min_duration = self.monitor.interval
        # Smooth IO with ~1s rolling window based on sampling interval
        try:
            self._io_window = max(
                1, int(round(1.0 / (self.monitor.interval or 1.0)))
            )
        except Exception:
            self._io_window = 1
        self._build_subsets()

    def _build_subsets(self):
        """Build a dictionary of metric subsets based on the provided
        configuration"""
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
                    "ylim": (0, self.monitor.gpu_memory),
                    "label": "GPU Memory (All GPUs)",
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
                    "ylim": (0, self.monitor.gpu_memory),
                    "label": "GPU Memory Summary",
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

    def _resolve_metric_subsets(
        self,
        metrics: Optional[List[str]]
    ) -> Tuple[str, ...]:
        """Map user-specified metrics or subsets to visualizer subset keys."""
        if not metrics:
            return ("cpu", "mem", "io")

        resolved: List[str] = []
        metric_list = (
            [metrics]
            if isinstance(metrics, str)
            else list(metrics)
        )
        for metric in metric_list:
            if not metric:
                continue
            metric_key = str(metric).strip()
            if metric_key in self.subsets:
                resolved.append(metric_key)
                continue
            found_subset = next(
                (
                    subset
                    for subset, cfg in self.subsets.items()
                    if metric_key in cfg
                ),
                None,
            )
            if found_subset:
                resolved.append(found_subset)
            else:
                logger.warning(
                    EXTENSION_ERROR_MESSAGES[
                        ExtensionErrorCode.INVALID_METRIC_SUBSET
                    ].format(
                        subset=metric_key,
                        supported_subsets=", ".join(self.subsets.keys()),
                    )
                )

        # Remove duplicates while preserving order; fall back to defaults
        deduped = tuple(dict.fromkeys(resolved))
        return deduped or ("cpu", "mem", "io")

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
                        "cell_index": cell["cell_index"],
                        "start_time": current_time,
                        "end_time": current_time + cell_duration,
                        "duration": cell_duration,
                    }
                )
                current_time += cell_duration

        return compressed_perfdata, cell_boundaries

    def _plot_direct(self, metric_subsets, cell_range, show_idle, level, save_jpeg=None, pickle_file=None):
        """Plot metrics directly with matplotlib without widgets"""
        start_idx, end_idx = cell_range
        filtered_cells = self.cell_history.view(start_idx, end_idx + 1)
        
        # Get performance data for the specified level
        perfdata = filter_perfdata(
            filtered_cells,
            self.monitor.data.view(level=level),
            not show_idle,
        )
        
        if perfdata.empty:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.NO_PERFORMANCE_DATA
                ]
            )
            return
        
        # Process time data
        if not show_idle:
            processed_data, self._compressed_cell_boundaries = (
                self._compress_time_axis(perfdata, cell_range)
            )
        else:
            processed_data = perfdata.copy()
            processed_data["time"] -= self.monitor.start_time
        
        # Get metrics for subsets
        metrics = []
        for subset in metric_subsets:
            if subset in self.subsets:
                for metric_key in self.subsets[subset].keys():
                    metrics.append(metric_key)
            else:
                logger.warning(
                    EXTENSION_ERROR_MESSAGES[
                        ExtensionErrorCode.INVALID_METRIC_SUBSET
                    ].format(
                        subset=subset,
                        supported_subsets=", ".join(self.subsets.keys()),
                    )
                )
        
        if not metrics:
            logger.warning("No valid metrics found to plot")
            return
        
        # Create subplots
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3 * n_metrics), 
                                constrained_layout=True)
        if n_metrics == 1:
            axes = [axes]
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            self._plot_metric(
                processed_data, metric, cell_range, show_idle, axes[i], level
            )
        
        # Handle JPEG saving
        if save_jpeg:
            if not save_jpeg.endswith('.jpg') and not save_jpeg.endswith('.jpeg'):
                save_jpeg += '.jpg'
            fig.savefig(save_jpeg, format='jpeg', dpi=300, bbox_inches='tight')
            print(f"Plot saved as JPEG: {save_jpeg}")
        
        # Handle pickle serialization
        if pickle_file:
            if not pickle_file.endswith('.pkl'):
                pickle_file += '.pkl'
            
            # Create plot data dictionary
            plot_data = {
                'figure': fig,
                'axes': axes,
                'metrics': metrics,
                'processed_data': processed_data,
                'cell_range': cell_range,
                'level': level,
                'show_idle': show_idle,
                'metric_subsets': metric_subsets
            }
            
            # Save to pickle file
            with open(pickle_file, 'wb') as f:
                pickle.dump(plot_data, f)
            
            # Print reload code
            print(f"Plot objects serialized to: {pickle_file}")
            print("\n# Python code to reload and display the plot:")
            print(f"import pickle")
            print(f"import matplotlib.pyplot as plt")
            print(f"")
            print(f"# Load the pickled plot data")
            print(f"with open('{pickle_file}', 'rb') as f:")
            print(f"    plot_data = pickle.load(f)")
            print(f"")
            print(f"# Extract the figure and display")
            print(f"fig = plot_data['figure']")
            print(f"plt.show()")
            print(f"")
            print(f"# Access other data:")
            print(f"# axes = plot_data['axes']")
            print(f"# metrics = plot_data['metrics']")
            print(f"# processed_data = plot_data['processed_data']")
            print(f"# cell_range = plot_data['cell_range']")
            print(f"# level = plot_data['level']")
        
        plt.show()

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
        self._draw_cell_boundaries(ax, cell_range, show_idle)

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
                    int(cell["cell_index"]),
                    0.4,
                )
        else:
            filtered_cells = self.cell_history.view()
            if cell_range:
                try:
                    mask = (filtered_cells["cell_index"] >= cell_range[0]) & (
                        filtered_cells["cell_index"] <= cell_range[1]
                    )
                    cells = filtered_cells[mask]
                except Exception:
                    cells = filtered_cells
            else:
                cells = filtered_cells
            for idx, cell in cells.iterrows():
                start_time = cell["start_time"] - self.monitor.start_time
                draw_cell_rect(
                    start_time, cell["duration"], int(cell["cell_index"]), 0.5
                )

    def plot(
        self,
        metric_subsets=("cpu", "mem", "io"),
        cell_range=None,
        show_idle=False,
        level=None,
        save_jpeg=None,
        pickle_file=None,
    ):
        metrics_missing = not metric_subsets
        if metrics_missing:
            metric_subsets = ("cpu", "mem", "io")
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
        try:
            min_cell_idx = int(valid_cells["cell_index"].min())
            max_cell_idx = int(valid_cells["cell_index"].max())
        except Exception:
            min_cell_idx, max_cell_idx = 0, len(valid_cells) - 1
        if cell_range is None:
            cell_start_index = 0
            for cell_idx in range(len(valid_cells) - 1, -1, -1):
                if valid_cells.iloc[cell_idx]["duration"] > self.min_duration:
                    cell_start_index = cell_idx
                    break
            start = int(valid_cells.iloc[cell_start_index]["cell_index"])
            end = int(valid_cells["cell_index"].max())
            if start > end:
                start, end = end, start
            cell_range = (start, end)

        # If level is specified, plot directly without widgets
        if level is not None:
            metric_subsets = self._resolve_metric_subsets(metric_subsets)
            return self._plot_direct(metric_subsets, cell_range, show_idle,
                                     level, save_jpeg, pickle_file)

        # Create interactive widgets
        style = {"description_width": "initial"}
        show_idle_checkbox = widgets.Checkbox(
            value=show_idle, description="Show idle periods"
        )
        # Sanitize slider value within bounds and ordered
        try:
            s0, s1 = cell_range
            if s0 > s1:
                s0, s1 = s1, s0
            s0 = max(min_cell_idx, min(s0, max_cell_idx))
            s1 = max(min_cell_idx, min(s1, max_cell_idx))
            slider_value = (s0, s1)
        except Exception:
            slider_value = (min_cell_idx, max_cell_idx)
        cell_range_slider = widgets.IntRangeSlider(
            value=slider_value,
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

        # Store the plot wrapper instance for persistent updates
        plot_wrapper = None

        def update_plots():
            nonlocal plot_wrapper
            current_cell_range, current_show_idle = (
                cell_range_slider.value,
                show_idle_checkbox.value,
            )
            start_idx, end_idx = current_cell_range
            cells_all = self.cell_history.view()
            try:
                mask = (cells_all["cell_index"] >= start_idx) & (
                    cells_all["cell_index"] <= end_idx
                )
                filtered_cells = cells_all[mask]
            except Exception:
                filtered_cells = cells_all
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
                if plot_wrapper is None:
                    # Create new plot wrapper only if it doesn't exist
                    plot_output.clear_output()
                    plot_wrapper = InteractivePlotWrapper(
                        self._plot_metric,
                        metrics,
                        labeled_options,
                        processed_perfdata,
                        current_cell_range,
                        current_show_idle,
                        self.figsize,
                    )
                    plot_wrapper.display_ui()
                else:
                    # Update existing plot wrapper with new data
                    plot_wrapper.update_data(
                        processed_perfdata,
                        current_cell_range,
                        current_show_idle,
                    )

        # Set up observers and display
        for widget in [show_idle_checkbox, cell_range_slider]:
            widget.observe(lambda change: update_plots(), names="value")

        display(widgets.VBox([config_widgets, plot_output]))
        update_plots()


class UnavailableVisualizer:
    """
    A stub that type-checks against VisualizerProtocol but
    only logs that visualization is unavailable.
    """
    def __init__(self, reason: str = "Plotting not available."):
        self._reason = reason

    def attach(self, monitor: MonitorProtocol) -> None: ...

    def plot(
        self,
        metric_subsets=("cpu", "mem", "io"),
        cell_range=None,
        show_idle=False,
    ) -> None:
        logger.info(
            EXTENSION_INFO_MESSAGES[ExtensionInfoCode.PLOTS_NOT_AVAILABLE].format(
                reason=self._reason
            )
        )



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
        figsize=None,
    ):
        self.plot_callback, self.perfdata_by_level, self.metrics = (
            plot_callback,
            perfdata_by_level,
            metrics,
        )
        self.labeled_options = labeled_options
        self.cell_range, self.show_idle, self.figsize = (
            cell_range,
            show_idle,
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

    def display_ui(self):
        """Display the Add button and all interactive panels."""
        display(widgets.VBox([self.add_panel_button, self.output_container]))
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
            plt.close(fig)
        output = widgets.Output()

        def update_plot():
            metric = metric_dropdown.value
            level = level_dropdown.value
            df = self.perfdata_by_level.get(level)
            if not is_ipympl_backend():
                output.clear_output(wait=True)
            with output:
                ax.clear()
                if df is not None and not df.empty:
                    self.plot_callback(
                        df, metric, self.cell_range, self.show_idle, ax, level
                    )
                fig.canvas.draw_idle()
                if not is_ipympl_backend():
                    display(fig)

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


def build_performance_visualizer(
    cell_history: CellHistory,
    plots_disabled: bool = False,
    plots_disabled_reason: str = "Plotting not available.",
) -> VisualizerProtocol:
    """
    Build PerformanceVisualizer object.
    Allows building a visualizer that degrades to a no-op when plots are disabled.
    """
    if plots_disabled:
        return UnavailableVisualizer(reason=plots_disabled_reason)
    return PerformanceVisualizer(cell_history)
