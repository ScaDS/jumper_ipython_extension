import argparse
import logging
from contextlib import contextmanager
from typing import Optional, Tuple, Union, List, Dict
import os
import sys
import time
import json
import zipfile
import tempfile
import shutil
from datetime import datetime

import pandas as pd

from jumper_extension.adapters.script_writer import NotebookScriptWriter
from jumper_extension.core.parsers import (
    parse_cell_range,
    parse_arguments,
    build_perfreport_parser,
    build_auto_perfreports_parser,
    build_export_perfdata_parser,
    build_export_cell_history_parser,
    build_import_perfdata_parser,
    build_import_cell_history_parser,
    build_export_session_parser,
    build_import_session_parser,
    ArgParsers,
)
from jumper_extension.core.state import Settings
from jumper_extension.core.messages import (
    ExtensionErrorCode,
    ExtensionInfoCode,
    EXTENSION_ERROR_MESSAGES,
    EXTENSION_INFO_MESSAGES,
)
from jumper_extension.adapters.monitor import PerformanceMonitorProtocol, PerformanceMonitor, OfflinePerformanceMonitor
from jumper_extension.adapters.session import SessionExporter, SessionImporter
from jumper_extension.adapters.visualizer import build_performance_visualizer, \
    PerformanceVisualizerProtocol
from jumper_extension.adapters.reporter import PerformanceReporter, build_performance_reporter
from jumper_extension.adapters.cell_history import CellHistory
from jumper_extension.utilities import get_available_levels


logger = logging.getLogger("extension")


class PerfmonitorService:
    """
    Core performance monitoring service with Python API.
    """
    def __init__(
        self,
        settings: Settings,
        monitor: PerformanceMonitorProtocol,
        visualizer: PerformanceVisualizerProtocol,
        reporter: PerformanceReporter,
        cell_history: CellHistory,
        script_writer: NotebookScriptWriter,
    ):
        self.settings = settings
        self.monitor = monitor
        self.visualizer = visualizer
        self.reporter = reporter
        self.cell_history = cell_history
        self.script_writer = script_writer
        self._skip_report = False

    def on_pre_run_cell(self, raw_cell: str, cell_magics: List[str], should_skip_report: bool):
        """Prepare for cell execution."""
        self.cell_history.start_cell(raw_cell, cell_magics)
        self._skip_report = should_skip_report

    def on_post_run_cell(self, result):
        """Handle post-cell execution, including automatic reports."""
        self.cell_history.end_cell(result)
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

    def show_resources(self):
        """Display available hardware resources (CPUs, memory, GPUs).

        Works for both live and imported sessions (when monitor.is_imported is True).
        """
        if not self.monitor.running and not self.monitor.is_imported:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_ACTIVE_MONITOR]
            )
            return
        if self.monitor.is_imported:
            logger.info(
                EXTENSION_INFO_MESSAGES[ExtensionInfoCode.IMPORTED_SESSION_RESOURCES].format(
                    source=self.monitor.session_source
                )
            )
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

    def show_cell_history(self):
        """Show interactive table of all executed cells."""
        self.cell_history.show_itable()

    def start_monitoring(self, interval: Optional[float] = None) -> Optional[ExtensionErrorCode]:
        """Start performance monitoring with specified interval.

        Args:
            interval: Monitoring interval in seconds (uses default if None)

        Returns:
            Error code if operation failed, None on success
        """
        # If an imported (offline) session is currently attached, swap to a live monitor
        if self.monitor.is_imported:
            self.monitor = PerformanceMonitor()

        if self.monitor.running:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.MONITOR_ALREADY_RUNNING]
            )
            return ExtensionErrorCode.MONITOR_ALREADY_RUNNING

        if interval is None:
            interval = self.settings.monitoring.default_interval
        else:
            self.settings.monitoring.user_interval = interval

        self.monitor.start(interval)
        self.settings.monitoring.running = self.monitor.running
        self.visualizer.attach(self.monitor)
        self.reporter.attach(self.monitor)
        return None

    def stop_monitoring(self):
        """Stop the active performance monitoring session."""
        if not self.monitor:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_ACTIVE_MONITOR]
            )
            return
        self.monitor.stop()
        self.settings.monitoring.running = False

    def plot_performance(self):
        """Open interactive plot for live or imported sessions."""
        if not self.monitor.running and not self.monitor.is_imported:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_ACTIVE_MONITOR]
            )
            return
        if self.monitor.is_imported:
            logger.info(
                EXTENSION_INFO_MESSAGES[ExtensionInfoCode.IMPORTED_SESSION_PLOT].format(
                    source=self.monitor.session_source
                )
            )
        self.visualizer.plot()

    def enable_perfreports(
        self,
        level: str,
        interval: Optional[float] = None,
        text: bool = False
    ):
        """Enable automatic performance reports after each cell execution.

        Args:
            level: Monitoring level (process, user, system, slurm)
            interval: Monitoring interval in seconds (starts monitoring if not running)
            text: Use text format instead of HTML
        """
        self.settings.perfreports.enabled = True
        self.settings.perfreports.level = level
        self.settings.perfreports.text = text

        format_message = "text" if text else "html"
        options_message = f"level: {level}, interval: {interval}, format: {format_message}"

        error_code = self.start_monitoring(interval)

        logger.info(
            EXTENSION_INFO_MESSAGES[
                ExtensionInfoCode.PERFORMANCE_REPORTS_ENABLED
            ].format(
                options_message=options_message,
            )
        )

    def disable_perfreports(self):
        """Disable automatic performance reports after cell execution."""
        self.settings.perfreports.enabled = False
        logger.info(
            EXTENSION_INFO_MESSAGES[
                ExtensionInfoCode.PERFORMANCE_REPORTS_DISABLED
            ]
        )

    def show_perfreport(
        self,
        cell_range: Optional[Tuple[int, int]] = None,
        level: Optional[str] = None,
        text: bool = False
    ):
        """Show performance report with optional cell range and level filters.

        Args:
            cell_range: Tuple of (start_idx, end_idx) or None for all cells
            level: Monitoring level or None to use default
            text: Use text format instead of HTML
        """
        if not self.monitor.running:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_ACTIVE_MONITOR]
            )
            return

        if text:
            self.reporter.print(cell_range=cell_range, level=level)
        else:
            self.reporter.display(cell_range=cell_range, level=level)

    def export_perfdata(
        self,
        file: Optional[str] = None,
        level: Optional[str] = None,
        name: Optional[str] = None
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """Export performance data to file or return as DataFrame.

        Args:
            file: File path for export or None to return DataFrame
            level: Monitoring level or None for default
            name: Custom variable name for returned DataFrame

        Returns:
            Dictionary with DataFrame if file is None, empty dict otherwise
        """
        if not self.monitor.running:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_ACTIVE_MONITOR]
            )
            return {}

        if file:
            self.monitor.data.export(
                file, level=level, cell_history=self.cell_history
            )
            return {}
        else:
            df = self.monitor.data.view(
                level=level, cell_history=self.cell_history
            )
            var_name = name or self.settings.export_vars.perfdata
            logger.info(
                EXTENSION_INFO_MESSAGES[
                    ExtensionInfoCode.PERFORMANCE_DATA_AVAILABLE
                ].format(var_name=var_name)
            )
            return {var_name: df}

    def load_perfdata(self, file: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Import performance data from file.

        Args:
            file: File path to import from

        Returns:
            Dictionary with loaded DataFrame
        """
        df = self.monitor.data.load(file)
        var_name = self.settings.loaded_vars.perfdata
        if df is not None:
            logger.info(
                EXTENSION_INFO_MESSAGES[
                    ExtensionInfoCode.PERFORMANCE_DATA_AVAILABLE
                ].format(var_name=var_name)
            )
        return {var_name: df}

    def export_cell_history(
        self,
        file: Optional[str] = None,
        name: Optional[str] = None
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """Export cell history to file or return as DataFrame.

        Args:
            file: File path for export or None to return DataFrame
            name: Custom variable name for returned DataFrame

        Returns:
            Dictionary with DataFrame if file is None, empty dict otherwise
        """
        if file:
            self.cell_history.export(file)
            return {}
        else:
            df = self.cell_history.view()
            var_name = name or self.settings.export_vars.cell_history
            logger.info(
                f"[JUmPER]: Cell history data available as '{var_name}'"
            )
            return {var_name: df}

    def load_cell_history(self, file: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Import cell history from file.

        Args:
            file: File path to import from

        Returns:
            Dictionary with loaded DataFrame
        """
        df = self.cell_history.load(file)
        var_name = self.settings.loaded_vars.cell_history
        if df is not None:
            logger.info(
                f"[JUmPER]: Cell history data available as '{var_name}'"
            )
        return {var_name: df}

    def export_session(self, path: Optional[str] = None) -> None:
        """Export full session using SessionExporter. If path ends with .zip, a zip is created automatically."""
        if not self.monitor.running and not self.monitor.is_imported:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_ACTIVE_MONITOR]
            )
        exporter = SessionExporter(self.monitor, self.cell_history, self.visualizer, self.reporter, logger)
        exporter.export(path)

    def import_session(self, path: str) -> None:
        """Import session using SessionImporter and announce success."""
        importer = SessionImporter(logger)
        ok = importer.import_(path, self)
        if ok:
            logger.info(
                EXTENSION_INFO_MESSAGES[ExtensionInfoCode.SESSION_IMPORTED].format(
                    source=self.monitor.session_source
                )
            )

    def fast_setup(self):
        """Quick setup: start perfmonitor and enable perfreports."""
        self.start_monitoring(1.0)
        self.enable_perfreports(level="process", interval=1.0, text=False)
        logger.info("[JUmPER]: Fast setup complete! Ready for interactive analysis.")

    def start_script_recording(self, output_path: Optional[str] = None):
        """Start recording code from cells to a Python script.

        Args:
            output_path: Optional output file path (auto-generated if None)
        """
        self.script_writer.start_recording(self.settings.snapshot(), output_path)

        if output_path:
            logger.info(f"[JUmPER]: Started script recording to '{output_path}'")
        else:
            logger.info("[JUmPER]: Started script recording (filename will be auto-generated)")

    def stop_script_recording(self) -> Optional[str]:
        """Stop recording and save accumulated code to file.

        Returns:
            Path to saved script file
        """
        if not self.script_writer:
            print("No script recording in progress.")
            return None

        output_path = self.script_writer.stop_recording()
        logger.info(f"Script saved to: {output_path}")
        return output_path

    @contextmanager
    def monitored(self):
        """Code performance monitoring context manager."""
        unavailable_message = "unavailable on monitored context"
        self.on_pre_run_cell(
            raw_cell=f"# <Code {unavailable_message}>",
            cell_magics=[f"<Magics {unavailable_message}>"],
            should_skip_report=False
        )
        try:
            yield self
        finally:
            self.on_post_run_cell(None)

    def close(self):
        """Close the magic_adapter and release any resources."""
        if self.monitor:
            self.monitor.stop()


class PerfmonitorMagicAdapter:
    """
    String-based adapter for IPython magic commands.
    Parses string arguments and delegates to PerfmonitorService.
    All methods must have the same names as magic commands to be recognized by script writer.
    """
    def __init__(
        self,
        service: PerfmonitorService,
        parsers: ArgParsers
    ):
        self.service = service
        self.parsers = parsers

    def on_pre_run_cell(self, raw_cell: str, cell_magics: List[str], should_skip_report: bool):
        """Delegate to magic_adapter."""
        self.service.on_pre_run_cell(raw_cell, cell_magics, should_skip_report)

    def on_post_run_cell(self, result):
        """Delegate to magic_adapter."""
        self.service.on_post_run_cell(result)

    def perfmonitor_resources(self, line: str):
        """Display available hardware resources (CPUs, memory, GPUs)."""
        self.service.show_resources()

    def show_cell_history(self, line: str):
        """Show interactive table of all executed cells with timestamps and durations."""
        self.service.show_cell_history()

    def perfmonitor_start(self, line: str):
        """Start performance monitoring with specified interval (default: 1 second)."""
        interval = None
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
        self.service.start_monitoring(interval)

    def perfmonitor_stop(self, line: str):
        """Stop the active performance monitoring session."""
        self.service.stop_monitoring()

    def perfmonitor_plot(self, line: str):
        """Open interactive plot with widgets for exploring performance data."""
        self.service.plot_performance()

    def perfmonitor_enable_perfreports(self, line: str):
        """Enable automatic performance reports after each cell execution."""
        args = parse_arguments(self.parsers.auto_perfreports, line)
        if args is None:
            return
        self.service.enable_perfreports(
            level=args.level,
            interval=float(args.interval) if args.interval else None,
            text=args.text
        )

    def perfmonitor_disable_perfreports(self, line: str):
        """Disable automatic performance reports after cell execution."""
        self.service.disable_perfreports()

    def perfmonitor_perfreport(self, line: str):
        """Show performance report with optional cell range and level filters."""
        args = parse_arguments(self.parsers.perfreport, line)
        if not args:
            return

        cell_range = None
        if args.cell:
            cell_range = self._parse_cell_range(args.cell)
            if cell_range is None:
                return

        self.service.show_perfreport(
            cell_range=cell_range,
            level=args.level,
            text=args.text
        )

    def perfmonitor_export_perfdata(self, line: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Export performance data or push as DataFrame."""
        args = parse_arguments(self.parsers.export_perfdata, line)
        return self.service.export_perfdata(
            file=args.file if args else None,
            level=args.level if args else None,
            name=args.name if args else None
        )

    def perfmonitor_load_perfdata(self, line: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Import performance data from file."""
        args = parse_arguments(self.parsers.import_perfdata, line)
        if not args:
            return {}
        return self.service.load_perfdata(args.file)

    def perfmonitor_export_cell_history(self, line: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Export cell history or push as DataFrame."""
        args = parse_arguments(self.parsers.export_cell_history, line)
        return self.service.export_cell_history(
            file=args.file if args else None,
            name=args.name if args else None
        )

    def perfmonitor_load_cell_history(self, line: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Import cell history from file."""
        args = parse_arguments(self.parsers.import_cell_history, line)
        if not args:
            return {}
        return self.service.load_cell_history(args.file)

    def perfmonitor_fast_setup(self, line: str):
        """Quick setup: start perfmonitor and enable perfreports."""
        self.service.fast_setup()

    def perfmonitor_help(self, line: str):
        """Show comprehensive help information for all available commands."""
        commands = [
            "perfmonitor_fast_setup -- quick setup: enable ipympl plots, start monitor, enable reports",
            "perfmonitor_help -- show this comprehensive help",
            "perfmonitor_resources -- show available hardware resources",
            "cell_history_show -- show interactive table of cell execution history",
            "perfmonitor_start [interval] -- start monitoring (default: 1 second)",
            "perfmonitor_stop -- stop monitoring",
            "perfmonitor_perfreport [--cell RANGE] [--level LEVEL] -- show report",
            "perfmonitor_plot -- interactive plot with widgets for data exploration",
            "perfmonitor_enable_perfreports [--level LEVEL] [--interval INTERVAL] [--text] -- enable auto-reports",
            "perfmonitor_disable_perfreports -- disable auto-reports",
            "perfmonitor_export_perfdata [--file FILE] [--level LEVEL] [--name NAME] -- export CSV;"
            " without --file pushes DataFrame (default 'perfdata_df')",
            "perfmonitor_export_cell_history [--file FILE] [--name NAME] -- export history to JSON/CSV;"
            " without --file pushes DataFrame (default 'cell_history_df')",
            "export_session [target|target.zip] -- export full session",
            "import_session <dir-or-zip> -- import full session for offline analysis",
        ]
        print("Available commands:")
        for cmd in commands:
            print(f"  {cmd}")

        print("\nMonitoring Levels:")
        print("  process -- current Python process only (default, most focused)")
        print("  user    -- all processes belonging to current user")
        print("  system  -- system-wide metrics across all processes")
        available_levels = get_available_levels()
        if "slurm" in available_levels:
            print("  slurm   -- processes within current SLURM job (HPC environments)")

        print("\nCell Range Formats:")
        print("  5       -- single cell (cell #5)")
        print("  2:8     -- range of cells (cells #2 through #8)")
        print("  :5      -- from start to cell #5")
        print("  3:      -- from cell #3 to end")

        print("\nMetric Categories:")
        print("  cpu, gpu, mem, io (default: all available)")
        print("  cpu_all, gpu_all for detailed per-core/per-GPU metrics")

    def export_session(self, line: str):
        """Export full session into a directory or zip.

        Usage:
          %export_session
          %export_session my_dir
          %export_session my.zip
        """
        args = parse_arguments(self.parsers.export_session, line)
        if args is None:
            return
        self.service.export_session(path=args.path)

    def import_session(self, line: str):
        """Import full session from a directory or zip.

        Usage:
          %import_session path/to/dir-or-zip
        """
        args = parse_arguments(self.parsers.import_session, line)
        if not args:
            return
        self.service.import_session(args.path)

    def start_write_script(self, line: str):
        """
        Start recording code from cells to a Python script.

        Usage:
          %start_write_script [output_path]

        Examples:
          %start_write_script
          %start_write_script my_script.py
        """
        output_path = line.strip() if line else None
        self.service.start_script_recording(output_path)

    def end_write_script(self, line: str):
        """Stop recording and save accumulated code to file."""
        self.service.stop_script_recording()

    def _parse_cell_range(self, cell_str: str) -> Optional[Tuple[int, int]]:
        """Parse a cell range string into start and end indices."""
        result = parse_cell_range(cell_str, len(self.service.cell_history))
        if result is None and cell_str:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.INVALID_CELL_RANGE
                ].format(cell_range=cell_str)
            )
        return result

    @contextmanager
    def monitored(self):
        """Code performance monitoring context manager."""
        with self.service.monitored():
            yield self

    def close(self):
        """Close the magic_adapter and release any resources."""
        self.service.close()


def build_perfmonitor_service(
        plots_disabled: bool = False,
        plots_disabled_reason: str = "Plotting not available.",
        display_disabled: bool = False,
        display_disabled_reason: str = "Display not available."
) -> PerfmonitorService:
    """Build a new instance of the perfmonitor magic_adapter (core API).

    Returns:
        PerfmonitorService instance with Python API
    """
    settings = Settings()
    monitor = PerformanceMonitor()
    cell_history = CellHistory()
    visualizer = build_performance_visualizer(
        cell_history,
        plots_disabled=plots_disabled,
        plots_disabled_reason=plots_disabled_reason,
    )
    reporter = build_performance_reporter(
        cell_history,
        display_disabled=display_disabled,
        display_disabled_reason=display_disabled_reason,
    )
    script_writer = NotebookScriptWriter(cell_history)

    return PerfmonitorService(
        settings=settings,
        monitor=monitor,
        visualizer=visualizer,
        reporter=reporter,
        cell_history=cell_history,
        script_writer=script_writer,
    )


def build_perfmonitor_magic_adapter(
        plots_disabled: bool = False,
        plots_disabled_reason: str = "Plotting not available.",
        display_disabled: bool = False,
        display_disabled_reason: str = "Display not available."
) -> PerfmonitorMagicAdapter:
    """Build a new instance of the perfmonitor magic adapter (string-based API).

    Returns:
        PerfmonitorMagicAdapter instance that wraps PerfmonitorService
    """
    service = build_perfmonitor_service(
        plots_disabled=plots_disabled,
        plots_disabled_reason=plots_disabled_reason,
        display_disabled=display_disabled,
        display_disabled_reason=display_disabled_reason,
    )

    parsers = ArgParsers(
        perfreport=build_perfreport_parser(),
        auto_perfreports=build_auto_perfreports_parser(),
        export_perfdata=build_export_perfdata_parser(),
        export_cell_history=build_export_cell_history_parser(),
        import_perfdata=build_import_perfdata_parser(),
        import_cell_history=build_import_cell_history_parser(),
        export_session=build_export_session_parser(),
        import_session=build_import_session_parser(),
    )

    return PerfmonitorMagicAdapter(
        service=service,
        parsers=parsers,
    )
