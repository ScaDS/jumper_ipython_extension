import argparse
import logging
import shlex
from typing import Union

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
from .utilities import get_available_levels
from .visualizer import PerformanceVisualizer

logger = logging.getLogger("extension")

_perfmonitor_magics = None


@magics_class
class perfmonitorMagics(Magics):
    def __init__(self, shell):
        super().__init__(shell)
        self.monitor = self.visualizer = self.reporter = None
        self.cell_history = CellHistory()
        self.print_perfreports = self._skip_report = False
        self.perfreports_level = "process"
        self.perfreports_text = False
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
            if self.perfreports_text:
                self.reporter.print(cell_range=None, level=self.perfreports_level)
            else:
                self.reporter.display(cell_range=None, level=self.perfreports_level)
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
        self._setup_performance_monitoring(line)


    def _setup_performance_monitoring(self, interval: Union[float, str]) -> Union[None, ExtensionErrorCode]:
        if self.monitor and self.monitor.running:
            return ExtensionErrorCode.MONITOR_ALREADY_RUNNING

        interval_number = 1.0
        if interval:
            try:
                interval_number = float(interval)
            except ValueError:
                return ExtensionErrorCode.INVALID_INTERVAL_VALUE

        self.monitor = PerformanceMonitor(interval=interval_number)
        self.monitor.start()
        self.visualizer = PerformanceVisualizer(
            self.monitor, self.cell_history, min_duration=interval_number
        )
        self.reporter = PerformanceReporter(
            self.monitor, self.cell_history, min_duration=interval_number
        )
        self.min_duration = interval
        return None

    @staticmethod
    def _handle_setup_error_messages(error_code: ExtensionErrorCode, interval: Union[float, str] = None):
        if error_code == ExtensionErrorCode.MONITOR_ALREADY_RUNNING:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.MONITOR_ALREADY_RUNNING
                ]
            )
        elif error_code == ExtensionErrorCode.INVALID_INTERVAL_VALUE:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.INVALID_INTERVAL_VALUE
                ].format(interval=interval)
            )


    @line_magic
    def perfmonitor_stop(self, line):
        """Stop the active performance monitoring session"""
        self._skip_report = True
        if not self.monitor:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_ACTIVE_MONITOR]
            )
            return
        self.monitor.stop()

    def _parse_arguments(self, line, parser: argparse.ArgumentParser = None):
        if not parser:
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
            "--text",
            action="store_true",
            help="Show report in text format"
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
        if not self.monitor:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_ACTIVE_MONITOR]
            )
            return
        self.visualizer.plot()

    @line_magic
    def perfmonitor_enable_perfreports(self, line):
        """Enable automatic performance reports after each cell execution"""
        self._skip_report = True
        self.print_perfreports = True

        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--interval",
            type=float,
            default=1.0,
            help="Interval between automatic reports (default: 1 second)",
        )
        args = self._parse_arguments(line, parser)
        if args is None:
            return

        self.perfreports_level = args.level
        self.perfreports_text = args.text
        interval = args.interval

        format_message = "text" if self.perfreports_text else "html"
        options_message = f"level: {self.perfreports_level}, interval: {interval}, format: {format_message}"

        error_code = self._setup_performance_monitoring(interval)
        self._handle_setup_error_messages(error_code, interval)

        logger.info(
            EXTENSION_INFO_MESSAGES[
                ExtensionInfoCode.PERFORMANCE_REPORTS_ENABLED
            ].format(
                options_message=options_message,
            )
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
        if args.text:
            self.reporter.print(cell_range=cell_range, level=args.level)
        else:
            self.reporter.display(cell_range=cell_range, level=args.level)

    @line_magic
    def perfmonitor_export_perfdata(self, line):
        """Export performance data or push as DataFrame

        Usage:
          %perfmonitor_export_perfdata --file <path> [--level LEVEL]
            # export to file
          %perfmonitor_export_perfdata [--level LEVEL]
            # push DataFrame
        """
        self._skip_report = True
        if not self.monitor:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_ACTIVE_MONITOR]
            )
            return

        # Parse optional --file and --level arguments
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--file", type=str, help="Output filename")
        parser.add_argument(
            "--level",
            default="process",
            choices=get_available_levels(),
            help="Performance level",
        )
        try:
            args = (
                parser.parse_args(shlex.split(line))
                if line
                else parser.parse_args([])
            )
        except Exception:
            args = None

        if args and args.file:
            self.monitor.data.export(
                args.file, level=args.level, cell_history=self.cell_history
            )
        else:
            df = self.monitor.data.view(
                level=args.level, cell_history=self.cell_history
            )
            var_name = "perfdata_df"
            self.shell.push({var_name: df})
            print(
                "[JUmPER]: Performance data DataFrame available as "
                f"'{var_name}'"
            )

    @line_magic
    def perfmonitor_export_cell_history(self, line):
        """Export cell history or push as DataFrame

        Usage:
          %perfmonitor_export_cell_history --file <path>  # export to file
          %perfmonitor_export_cell_history                # push DataFrame
        """
        self._skip_report = True

        # Parse optional --file argument
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--file", type=str, help="Output filename")
        try:
            args = parser.parse_args(shlex.split(line)) if line else None
        except Exception:
            args = None

        if args and args.file:
            self.cell_history.export(args.file)
        else:
            df = self.cell_history.view()
            var_name = "cell_history_df"
            self.shell.push({var_name: df})
            print(
                f"[JUmPER]: Cell history DataFrame available as '{var_name}'"
            )

    @line_magic
    def perfmonitor_fast_setup(self, line):
        """Quick setup: enable ipympl interactive plots, start perfmonitor, and enable perfreports"""
        self._skip_report = True
        
        # 1. Enable ipympl interactive plots
        try:
            self.shell.run_line_magic('matplotlib', 'ipympl')
            print("[JUmPER]: Enabled ipympl interactive plots")
        except Exception as e:
            logger.warning(f"Failed to enable ipympl interactive plots: {e}")
        
        # 2. Start performance monitor with default interval (1 second)
        self.perfmonitor_start("1.0")
        
        # 3. Enable performance reports with default level (process)
        self.perfmonitor_enable_perfreports("--level process")
        
        print("[JUmPER]: Fast setup complete! Ready for interactive analysis.")

    @line_magic
    def perfmonitor_help(self, line):
        """Show comprehensive help information for all available commands"""
        self._skip_report = True
        commands = [
            "perfmonitor_fast_setup -- quick setup: enable ipympl plots, start monitor, enable reports",
            "perfmonitor_help -- show this comprehensive help",
            "perfmonitor_resources -- show available hardware resources",
            "cell_history -- show interactive table of cell execution history",
            "perfmonitor_start [interval] -- start monitoring "
            "(default: 1 second)",
            "perfmonitor_stop -- stop monitoring",
            "perfmonitor_perfreport [--cell RANGE] [--level LEVEL] -- "
            "show report",
            "perfmonitor_plot -- interactive plot with widgets for data "
            "exploration",
            "perfmonitor_enable_perfreports [--level LEVEL] [--interval INTERVAL] [--text] -- enable "
            "auto-reports",
            "perfmonitor_disable_perfreports -- disable auto-reports",
            "perfmonitor_export_perfdata [--file FILE] [--level LEVEL] -- "
            "export CSV; without --file pushes DataFrame "
            "'perfdata_df'",
            "perfmonitor_export_cell_history [--file FILE] -- export "
            "history to JSON/CSV; without --file pushes DataFrame "
            "'cell_history_df'",
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
