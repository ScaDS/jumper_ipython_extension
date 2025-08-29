import argparse
import logging
import os
import shlex

from IPython.core.magic import Magics, line_magic, magics_class

from .cell_history import CellHistory
from .extension_messages import (
    ExtensionErrorCode,
    ExtensionInfoCode,
    EXTENSION_ERROR_MESSAGES,
    EXTENSION_INFO_MESSAGES,
)
from .monitor import PerformanceMonitor
from .reporter import PerformanceReporter
from .utilities import (
    get_available_levels,
    save_perfdata_to_disk,
    save_cell_history_to_disk,
    save_monitor_metadata_to_disk,
    load_perfdata_from_disk,
    load_cell_history_from_disk,
    load_monitor_metadata_from_disk,
)
from .visualizer import PerformanceVisualizer
from .bali_adapter import BaliMagicsMixin

logger = logging.getLogger("extension")

_perfmonitor_magics = None


@magics_class
class perfmonitorMagics(Magics, BaliMagicsMixin):
    def __init__(self, shell):
        Magics.__init__(self, shell)
        BaliMagicsMixin.__init__(self)
        self.monitor = self.visualizer = self.reporter = None
        self.cell_history = CellHistory()
        self.print_perfreports = self._skip_report = False
        self.perfreports_level = "process"
        self.min_duration = None

    def pre_run_cell(self, info):
        self.cell_history.start_cell(info.raw_cell)
        self._skip_report = False

    def post_run_cell(self, result):
        self.cell_history.end_cell(result.result)
        if (
            self.monitor
            and self.reporter
            and self.print_perfreports
            and not self._skip_report
        ):
            self.reporter.print(cell_range=None, level=self.perfreports_level)
        self._skip_report = False

    @line_magic
    def perfmonitor_resources(self, line):
        """Display available hardware resources (CPUs, memory, GPUs)"""
        self._skip_report = True
        if not self.monitor:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_ACTIVE_MONITOR]
            )
            return
        print("[JUmPER]:")
        cpu_info = (
            f"  CPUs: {self.monitor.num_cpus}\n    "
            f"CPU affinity: {self.monitor.cpu_handles}"
        )
        print(cpu_info)
        mem_gpu_info = (
            f"  Memory: {self.monitor.memory_limits['system']} GB\n  "
            f"GPUs: {self.monitor.num_gpus}"
        )
        print(mem_gpu_info)
        if self.monitor.num_gpus:
            print(f"    {self.monitor.gpu_name}, {self.monitor.gpu_memory} GB")

    @line_magic
    def cell_history(self, line):
        """Show interactive table of all executed cells with timestamps and
        durations"""
        self._skip_report = True
        self.cell_history.show_itable()

    @line_magic
    def perfmonitor_start(self, line):
        """Start performance monitoring with specified interval
        (default: 1 second)"""
        self._skip_report = True
        if self.monitor and self.monitor.running:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.MONITOR_ALREADY_RUNNING
                ]
            )
            return

        interval = 1.0
        if line:
            try:
                interval = float(line)
            except ValueError:
                logger.warning(
                    EXTENSION_ERROR_MESSAGES[
                        ExtensionErrorCode.INVALID_INTERVAL_VALUE
                    ].format(interval=line)
                )
                return

        self.monitor = PerformanceMonitor(interval=interval)
        self.monitor.start()
        self.visualizer = PerformanceVisualizer(
            self.monitor, self.cell_history, min_duration=interval, bali_adapter=self.bali_adapter
        )
        self.reporter = PerformanceReporter(
            self.monitor, self.cell_history, min_duration=interval
        )
        self.min_duration = interval

    @line_magic
    def perfmonitor_stop(self, line):
        """Stop the active performance monitoring session"""
        self._skip_report = True
        if not self.monitor:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_ACTIVE_MONITOR]
            )
            return
        
        # Save data to disk before stopping
        save_perfdata_to_disk(self.monitor.pid, self.monitor.data)
        save_cell_history_to_disk(self.monitor.pid, self.cell_history)
        save_monitor_metadata_to_disk(self.monitor.pid, self.monitor)
        
        self.monitor.stop()

    def _parse_arguments(self, line):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--cell", type=str, help="Cell index or range (e.g., 5, 2:8, :5)"
        )
        parser.add_argument(
            "--level",
            default="process",
            choices=get_available_levels(),
            help="Performance level",
        )
        parser.add_argument(
            "--from-disk", 
            action="store_true", 
            help="Load data from disk instead of memory"
        )
        try:
            return parser.parse_args(shlex.split(line))
        except Exception:
            return None

    def _parse_cell_range(self, cell_str, cell_history):
        if not cell_str:
            return None
        try:
            max_idx = len(cell_history) - 1
            if ":" in cell_str:
                start_str, end_str = cell_str.split(":", 1)
                start_idx = 0 if not start_str else int(start_str)
                end_idx = max_idx if not end_str else int(end_str)
            else:
                start_idx = end_idx = int(cell_str)
            if 0 <= start_idx <= end_idx <= max_idx:
                return (start_idx, end_idx)
        except (ValueError, IndexError):
            logger.warning(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.INVALID_CELL_RANGE
                ].format(cell_range=cell_str)
            )
        return None

    @line_magic
    def perfmonitor_plot(self, line):
        """Open interactive plot with widgets for exploring performance data"""
        self._skip_report = True
        
        # Parse arguments for --from-disk option
        args = self._parse_arguments(line)
        from_disk = args and getattr(args, 'from_disk', False)
        
        if from_disk:
            # Load data from disk
            pid = self.monitor.pid if self.monitor else os.getpid()
            cell_history_data = load_cell_history_from_disk(pid)
            if cell_history_data.empty:
                logger.warning(f"No cell history found on disk for PID {pid}")
                return
            
            # Create a temporary cell history object
            from .cell_history import CellHistory
            temp_cell_history = CellHistory()
            temp_cell_history.data = cell_history_data
            
            # Create a mock visualizer with disk data
            from .visualizer import DiskPerformanceVisualizer
            visualizer = DiskPerformanceVisualizer(pid, temp_cell_history)
            visualizer.plot()
        else:
            # Use memory data (original behavior)
            if not self.monitor:
                logger.warning(
                    EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_ACTIVE_MONITOR]
                )
                return
            self.visualizer.plot()

    @line_magic
    def bali_segments(self, line):
        """Show interactive table of all BALI segments with start/end times"""
        self._skip_report = True
        self._bali_segments(line)

    # _bali_refresh_from_disk method is now provided by BaliMagicsMixin

    @line_magic
    def bali_run(self, line):
        """Refresh persistent BALI segments from latest results on disk."""
        self._skip_report = True
        self._bali_run(line)

    @line_magic
    def perfmonitor_enable_perfreports(self, line):
        """Enable automatic performance reports after each cell execution"""
        self._skip_report = True
        args = self._parse_arguments(line)
        if args is None:
            return
        self.perfreports_level = args.level
        self.print_perfreports = True
        logger.info(
            EXTENSION_INFO_MESSAGES[
                ExtensionInfoCode.PERFORMANCE_REPORTS_ENABLED
            ].format(level=self.perfreports_level)
        )

    @line_magic
    def perfmonitor_disable_perfreports(self, line):
        """Disable automatic performance reports after cell execution"""
        self._skip_report = True
        self.print_perfreports = False
        logger.info(
            EXTENSION_INFO_MESSAGES[
                ExtensionInfoCode.PERFORMANCE_REPORTS_DISABLED
            ]
        )

    @line_magic
    def perfmonitor_perfreport(self, line):
        """Show performance report with optional cell range and level
        filters"""
        self._skip_report = True
        if not self.reporter:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_ACTIVE_MONITOR]
            )
            return
        args = self._parse_arguments(line)
        if not args:
            return
        cell_range = None
        if args.cell:
            cell_range = self._parse_cell_range(args.cell, self.cell_history)
            if not cell_range:
                return
        self.reporter.print(cell_range=cell_range, level=args.level)

    @line_magic
    def perfmonitor_export_perfdata(self, line):
        """Export performance data to CSV with specified monitoring level"""
        self._skip_report = True
        if not self.monitor:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_ACTIVE_MONITOR]
            )
            return
        parts = line.strip().split()
        filename = "performance_data.csv"
        if parts and not parts[0].startswith("--"):
            filename = parts[0]
            line = " ".join(parts[1:])
        args = self._parse_arguments(line)
        if not args:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.DEFINE_LEVEL
                ].format(levels=", ".join(get_available_levels()))
            )
            return
        self.monitor.data.export(filename, level=args.level)
        logger.info(
            EXTENSION_INFO_MESSAGES[ExtensionInfoCode.EXPORT_SUCCESS].format(
                filename=filename
            )
        )

    @line_magic
    def perfmonitor_perfdata_to_dataframe(self, line):
        """Export performance data to dataframe with specified monitoring
        level"""
        self._skip_report = True
        if not self.monitor:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_ACTIVE_MONITOR]
            )
            return
        parts = line.strip().split()

        dataframe_name = None
        if parts and not parts[0].startswith("--"):
            dataframe_name = parts[0]
            line = " ".join(parts[1:])
        args = self._parse_arguments(line)
        if not args:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.DEFINE_LEVEL
                ].format(levels=", ".join(get_available_levels()))
            )
            return
        dataframe_value = self.monitor.data.view(level=args.level)
        self.shell.push({dataframe_name: dataframe_value})
        logger.info(
            EXTENSION_INFO_MESSAGES[ExtensionInfoCode.EXPORT_SUCCESS].format(
                filename=dataframe_name
            )
        )

    @line_magic
    def perfmonitor_export_cell_history(self, line):
        """Export cell history to JSON or CSV format"""
        self._skip_report = True
        self.cell_history.export(line.strip() or "cell_history.json")

    @line_magic
    def perfmonitor_help(self, line):
        """Show comprehensive help information for all available commands"""
        self._skip_report = True
        commands = [
            "perfmonitor_help -- show this comprehensive help",
            "perfmonitor_resources -- show available hardware resources",
            "cell_history -- show interactive table of cell execution history",
            "bali_segments -- show interactive table of BALI segments",
            "bali_run -- refresh BALI segments from disk",
            "perfmonitor_start [interval] -- start monitoring "
            "(default: 1 second)",
            "perfmonitor_stop -- stop monitoring",
            "perfmonitor_perfreport [--cell RANGE] [--level LEVEL] -- "
            "show report",
            "perfmonitor_plot [--from-disk] -- interactive plot with widgets for data "
            "exploration",
            "perfmonitor_enable_perfreports [--level LEVEL] -- enable "
            "auto-reports",
            "perfmonitor_disable_perfreports -- disable auto-reports",
            "perfmonitor_export_perfdata [filename] [--level LEVEL] -- "
            "export CSV",
            "perfmonitor_export_cell_history [filename] -- export history to "
            "JSON/CSV",
            "perfmonitor_perfdata_to_dataframe [df_name] [--level LEVEL] -- "
            "perfdata to dataframe",
        ]
        print("Available commands:")
        for cmd in commands:
            print(f"  {cmd}")

        print("\nMonitoring Levels:")
        print(
            "  process -- current Python process only (default, most focused)"
        )
        print("  user    -- all processes belonging to current user")
        print("  system  -- system-wide metrics across all processes")
        available_levels = get_available_levels()
        if "slurm" in available_levels:
            print(
                "  slurm   -- processes within current SLURM job "
                "(HPC environments)"
            )

        print("\nCell Range Formats:")
        print("  5       -- single cell (cell #5)")
        print("  2:8     -- range of cells (cells #2 through #8)")
        print("  :5      -- from start to cell #5")
        print("  3:      -- from cell #3 to end")

        print("\nMetric Categories:")
        print("  cpu, gpu, mem, io (default: all available)")
        print("  cpu_all, gpu_all for detailed per-core/per-GPU metrics")


def load_ipython_extension(ipython):
    global _perfmonitor_magics
    _perfmonitor_magics = perfmonitorMagics(ipython)
    ipython.events.register("pre_run_cell", _perfmonitor_magics.pre_run_cell)
    ipython.events.register("post_run_cell", _perfmonitor_magics.post_run_cell)
    ipython.register_magics(_perfmonitor_magics)
    logger.info(EXTENSION_INFO_MESSAGES[ExtensionInfoCode.EXTENSION_LOADED])


def unload_ipython_extension(ipython):
    global _perfmonitor_magics
    if _perfmonitor_magics:
        ipython.events.unregister(
            "pre_run_cell", _perfmonitor_magics.pre_run_cell
        )
        ipython.events.unregister(
            "post_run_cell", _perfmonitor_magics.post_run_cell
        )
        if _perfmonitor_magics.monitor:
            _perfmonitor_magics.monitor.stop()
        _perfmonitor_magics = None
