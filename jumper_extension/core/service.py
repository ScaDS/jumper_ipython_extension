import argparse
import logging
import shlex
from dataclasses import dataclass
from typing import Optional, Tuple, Union

from jumper_extension.core.state import ExtensionState
from jumper_extension.core.messages import (
    ExtensionErrorCode,
    ExtensionInfoCode,
    EXTENSION_ERROR_MESSAGES,
    EXTENSION_INFO_MESSAGES,
)
from jumper_extension.adapters.monitor import PerformanceMonitor
from jumper_extension.adapters.visualizer import PerformanceVisualizer
from jumper_extension.adapters.reporter import PerformanceReporter
from jumper_extension.adapters.cell_history import CellHistory
from jumper_extension.utilities import is_pure_line_magic_cell, get_available_levels


logger = logging.getLogger("extension")


class PerfmonitorService:
    """Application service that encapsulates all extension logic."""
    def __init__(
        self,
        state: ExtensionState,
        monitor: PerformanceMonitor,
        visualizer: PerformanceVisualizer,
        reporter: PerformanceReporter,
        cell_history: CellHistory,
        shell_push_cb,     # callable for pushing variables into user ns
    ):
        self.state = state
        self.monitor = monitor
        self.visualizer = visualizer
        self.reporter = reporter
        self.cell_history = cell_history
        self.script_writer = None
        self.shell_push = shell_push_cb

    def pre_run_cell(self, info):
        self.cell_history.start_cell(info.raw_cell)
        self._skip_report = is_pure_line_magic_cell(info.raw_cell)

    def post_run_cell(self, result):
        self.cell_history.end_cell(result.result)
        if (
                not self._skip_report
                and self.state.runtime.monitor_is_running
                and self.state.settings.perfreports_enabled
        ):
            if self.state.settings.perfreports_text:
                self.reporter.print(
                    cell_range=None, level=self.state.settings.perfreports_level
                )
            else:
                self.reporter.display(
                    cell_range=None, level=self.state.settings.perfreports_level
                    )


    def perfmonitor_resources(self, line):
        """Display available hardware resources (CPUs, memory, GPUs)"""
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


    def cell_history(self, line):
        """Show interactive table of all executed cells with timestamps and
        durations"""
        self.cell_history.show_itable()


    def perfmonitor_start(self, line):
        """Start performance monitoring with specified interval
        (default: 1 second)"""
        self._setup_performance_monitoring(line)

    def _setup_performance_monitoring(self, interval: Union[float, str]) -> Union[None, ExtensionErrorCode]:
        if self.state.runtime.monitor_is_running:
            return ExtensionErrorCode.MONITOR_ALREADY_RUNNING

        if interval:
            try:
                interval_number = float(interval)
            except ValueError:
                return ExtensionErrorCode.INVALID_INTERVAL_VALUE
        else:
            interval_number = self.state.settings.default_interval

        self.monitor = PerformanceMonitor(interval=interval_number)
        self.monitor.start()
        self.state.mark_monitor_running(interval_number, self.monitor.start_time)

        self.visualizer = PerformanceVisualizer(
            self.monitor, self.cell_history, min_duration=interval_number
        )
        self.reporter = PerformanceReporter(
            self.monitor, self.cell_history, min_duration=interval_number
        )
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

    def perfmonitor_stop(self, line):
        """Stop the active performance monitoring session"""
        if not self.monitor:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_ACTIVE_MONITOR]
            )
            return
        self.monitor.stop()
        self.state.mark_monitor_stopped(self.monitor.stop_time)

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

    def perfmonitor_plot(self, line):
        """Open interactive plot with widgets for exploring performance data"""
        if not self.state.runtime.monitor_is_running:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_ACTIVE_MONITOR]
            )
            return
        self.visualizer.plot()

    def perfmonitor_enable_perfreports(self, line):
        """Enable automatic performance reports after each cell execution"""
        self.state.settings.perfreports_enabled = True

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

        self.state.settings.perfreports_level = args.level
        self.state.settings.perfreports_text = args.text
        interval = args.interval

        format_message = "text" if self.state.settings.perfreports_text else "html"
        options_message = (f"level: {self.state.settings.perfreports_level},"
                           f" interval: {interval}, format: {format_message}")

        error_code = self._setup_performance_monitoring(interval)
        self._handle_setup_error_messages(error_code, interval)

        logger.info(
            EXTENSION_INFO_MESSAGES[
                ExtensionInfoCode.PERFORMANCE_REPORTS_ENABLED
            ].format(
                options_message=options_message,
            )
        )

    def perfmonitor_disable_perfreports(self, line):
        """Disable automatic performance reports after cell execution"""
        self.state.settings.perfreports_enabled = False
        logger.info(
            EXTENSION_INFO_MESSAGES[
                ExtensionInfoCode.PERFORMANCE_REPORTS_DISABLED
            ]
        )

    def perfmonitor_perfreport(self, line):
        """Show performance report with optional cell range and level
        filters"""
        if not self.state.runtime.monitor_is_running:
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

    def perfmonitor_export_perfdata(self, line):
        """Export performance data or push as DataFrame

        Usage:
          %perfmonitor_export_perfdata --file <path> [--level LEVEL]
            # export to file
          %perfmonitor_export_perfdata [--level LEVEL]
            # push DataFrame
        """
        if not self.state.runtime.monitor_is_running:
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

    def perfmonitor_export_cell_history(self, line):
        """Export cell history or push as DataFrame

        Usage:
          %perfmonitor_export_cell_history --file <path>  # export to file
          %perfmonitor_export_cell_history                # push DataFrame
        """

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

    def perfmonitor_fast_setup(self, line):
        """Quick setup: enable ipympl interactive plots, start perfmonitor, and enable perfreports"""
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

    def perfmonitor_help(self, line):
        """Show comprehensive help information for all available commands"""
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

    def start_write_script(self, line):
        """
        Start recording code from cells to a Python script.

        Usage:
          %start_write_script [output_path]

        Examples:
          %start_write_script
          %start_write_script my_script.py
        """
        output_path = line.strip() if line else None
        self.script_writer.start_recording(output_path)

        if output_path:
            print(f"[JUmPER]: Started script recording to '{output_path}'")
        else:
            print("[JUmPER]: Started script recording (filename will be auto-generated)")

    def end_write_script(self, line):
        """
        Stop recording and save accumulated code to file.

        Usage:
          %end_write_script
        """
        output_path = self.script_writer.stop_recording()

        if output_path:
            print(f"[JUmPER]: Script successfully saved to '{output_path}'")
        else:
            print("[JUmPER]: Failed to save script (recording may not have been active)")