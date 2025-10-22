import argparse
import logging
from typing import Optional, Tuple, Union, List, Dict

import pandas as pd

from jumper_extension.core.parsers import parse_cell_range, parse_arguments, build_perfreport_parser, \
    build_auto_perfreports_parser, build_export_perfdata_parser, build_export_cell_history_parser, Parsers

from jumper_extension.core.state import UserSettings
from jumper_extension.core.messages import (
    ExtensionErrorCode,
    ExtensionInfoCode,
    EXTENSION_ERROR_MESSAGES,
    EXTENSION_INFO_MESSAGES,
)
from jumper_extension.adapters.monitor import PerformanceMonitorProtocol, PerformanceMonitor
from jumper_extension.adapters.visualizer import PerformanceVisualizer
from jumper_extension.adapters.reporter import PerformanceReporter
from jumper_extension.adapters.cell_history import CellHistory
from jumper_extension.utilities import get_available_levels


logger = logging.getLogger("extension")


class PerfmonitorService:
    """
    Application service that encapsulates all extension logic.
    All methods must have the same names as magic commands to be recognized by script writer.
    """
    def __init__(
        self,
        settings: UserSettings,
        monitor: PerformanceMonitorProtocol,
        visualizer: PerformanceVisualizer,
        reporter: PerformanceReporter,
        cell_history: CellHistory,
        parsers: Parsers
    ):
        self.settings = settings
        self.monitor = monitor
        self.visualizer = visualizer
        self.reporter = reporter
        self.cell_history = cell_history
        self.script_writer = None
        self._skip_report = False
        self.parsers = parsers


    def on_pre_run_cell(self, raw_cell: str, cell_magics: List[str], should_skip_report: bool):
        self.cell_history.start_cell(raw_cell, cell_magics)
        self._skip_report = should_skip_report

    def on_post_run_cell(self, result):
        self.cell_history.end_cell(result.result)
        if (
                not self._skip_report
                and self.monitor.running
                and self.settings.perfreports.enabled
        ):
            if self.settings.perfreports.text:
                self.reporter.print(
                    cell_range=None, level=self.settings.perfreports.level
                )
            else:
                self.reporter.display(
                    cell_range=None, level=self.settings.perfreports.level
                    )

    def perfmonitor_resources(self):
        """Display available hardware resources (CPUs, memory, GPUs)"""
        if not self.monitor.running:
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

    def cell_history(self):
        """Show interactive table of all executed cells with timestamps and
        durations"""
        self.cell_history.show_itable()

    def perfmonitor_start(self, line):
        """Start performance monitoring with specified interval
        (default: 1 second)"""
        error_code = self._setup_performance_monitoring(line)
        self._handle_setup_error_messages(error_code, line)

    def _setup_performance_monitoring(self, interval: Union[float, str]) -> Union[None, ExtensionErrorCode]:
        if self.monitor.running:
            return ExtensionErrorCode.MONITOR_ALREADY_RUNNING

        if interval:
            try:
                interval_number = float(interval)
                self.settings.user_interval = interval_number
            except ValueError:
                return ExtensionErrorCode.INVALID_INTERVAL_VALUE
        else:
            interval_number = self.settings.default_interval

        self.monitor.start(interval_number)
        self.visualizer.attach(self.monitor)
        self.reporter.attach(self.monitor)
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

    def perfmonitor_stop(self):
        """Stop the active performance monitoring session"""
        if not self.monitor:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_ACTIVE_MONITOR]
            )
            return
        self.monitor.stop()

    def _parse_cell_range(self, cell_str, cell_history) -> Optional[Tuple[int, int]]:
        """Parse a cell range string into start and end indices.
        
        Args:
            cell_str: String representing cell range (e.g., "1:3", "5", ":10")
            cell_history: List of cell history entries to validate indices against
            
        Returns:
            Tuple of (start_idx, end_idx) or None if invalid
        """
        result = parse_cell_range(cell_str, cell_history)
        if result is None and cell_str:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.INVALID_CELL_RANGE
                ].format(cell_range=cell_str)
            )
        return result

    def perfmonitor_plot(self):
        """Open interactive plot with widgets for exploring performance data"""
        if not self.monitor.running:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_ACTIVE_MONITOR]
            )
            return
        self.visualizer.plot()

    def perfmonitor_enable_perfreports(self, line):
        """Enable automatic performance reports after each cell execution"""
        self.settings.perfreports.enabled = True

        args = parse_arguments(self.parsers.auto_perfreports, line)
        if args is None:
            return

        self.settings.perfreports.level = args.level
        self.settings.perfreports.text = args.text
        interval = args.interval

        format_message = "text" if self.settings.perfreports.text else "html"
        options_message = (f"level: {self.settings.perfreports.level},"
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

    def perfmonitor_disable_perfreports(self):
        """Disable automatic performance reports after cell execution"""
        self.settings.perfreports.enabled = False
        logger.info(
            EXTENSION_INFO_MESSAGES[
                ExtensionInfoCode.PERFORMANCE_REPORTS_DISABLED
            ]
        )

    def perfmonitor_perfreport(self, line):
        """Show performance report with optional cell range and level
        filters"""
        if not self.monitor.running:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_ACTIVE_MONITOR]
            )
            return
        args = parse_arguments(self.parsers.perfreport, line)
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

    def perfmonitor_export_perfdata(self, line) -> Optional[Dict[str, pd.DataFrame]]:
        """Export performance data or push as DataFrame

        Usage:
          %perfmonitor_export_perfdata --file <path> [--level LEVEL]
            # export to file
          %perfmonitor_export_perfdata [--level LEVEL]
            # push DataFrame
        """
        if not self.monitor.running:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_ACTIVE_MONITOR]
            )
            return {}

        # Parse optional --file and --level arguments
        args = parse_arguments(self.parsers.export_perfdata, line)

        if args and args.file:
            self.monitor.data.export(
                args.file, level=args.level, cell_history=self.cell_history
            )
            return {}
        else:
            df = self.monitor.data.view(
                level=args.level, cell_history=self.cell_history
            )
            var_name = self.settings.export_vars.perfdata
            logger.info(
                EXTENSION_INFO_MESSAGES[
                    ExtensionInfoCode.PERFORMANCE_DATA_AVAILABLE
                ].format(var_name=var_name)
            )
            return {var_name: df}

    def perfmonitor_export_cell_history(self, line):
        """Export cell history or push as DataFrame

        Usage:
          %perfmonitor_export_cell_history --file <path>  # export to file
          %perfmonitor_export_cell_history                # push DataFrame
        """

        # Parse optional --file argument
        args = parse_arguments(self.parsers.export_cell_history, line)
        if args and args.file:
            self.cell_history.export(args.file)
            return {}
        else:
            df = self.cell_history.view()
            var_name = "cell_history_df"
            print(
                f"[JUmPER]: Cell history DataFrame available as '{var_name}'"
            )
            return {var_name: df}

    def perfmonitor_fast_setup(self):
        """Quick setup: start perfmonitor, and enable perfreports"""
        # 2. Start performance monitor with default interval (1 second)
        self.perfmonitor_start("1.0")

        # 3. Enable performance reports with default level (process)
        self.perfmonitor_enable_perfreports("--level process")

        print("[JUmPER]: Fast setup complete! Ready for interactive analysis.")

    def perfmonitor_help(self):
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
        if not self.script_writer:
            print("No script recording in progress. Use %start_write_script first.")
            return

        output_path = self.script_writer.stop()
        self.script_writer = None
        print(f"Script saved to: {output_path}")


    def close(self):
        """Close the service and release any resources."""
        if self.monitor:
            self.monitor.stop()


def build_perfmonitor_service():
    """Build a new instance of the perfmonitor service."""
    settings = UserSettings()
    monitor = PerformanceMonitor()
    cell_history = CellHistory()
    visualizer = PerformanceVisualizer(cell_history)
    reporter = PerformanceReporter(cell_history)
    parsers = Parsers(
        perfreport=build_perfreport_parser(),
        auto_perfreports=build_auto_perfreports_parser(),
        export_perfdata=build_export_perfdata_parser(),
        export_cell_history=build_export_cell_history_parser(),
    )
    return PerfmonitorService(
        settings,
        monitor,
        visualizer,
        reporter,
        cell_history,
        parsers,
    )
