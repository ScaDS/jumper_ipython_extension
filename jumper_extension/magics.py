import argparse
import shlex

from IPython.core.magic import Magics, line_magic, magics_class

from .cell_history import CellHistory
from .monitor import PerformanceMonitor
from .reporter import PerformanceReporter
from .utilities import get_available_levels
from .visualizer import PerformanceVisualizer

_perfmonitor_magics = None


@magics_class
class perfmonitorMagics(Magics):
    def __init__(self, shell):
        super().__init__(shell)
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
            return print("[JUmPER]: No active performance monitoring session")
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
            return print("[JUmPER]: Performance monitoring already running")

        interval = 1.0
        if line:
            try:
                interval = float(line)
            except ValueError:
                return print(f"[JUmPER]: Invalid interval value: {line}")

        self.monitor = PerformanceMonitor(interval=interval)
        self.monitor.start()
        self.visualizer = PerformanceVisualizer(
            self.monitor, self.cell_history, min_duration=interval
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
            print("[JUmPER]: No active performance monitoring session")
            return
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
            error_msg = (
                f"[JUmPER]: Invalid cell range: {cell_str} "
                f"(valid range: 0-{max_idx})"
            )
            print(error_msg)
        except (ValueError, IndexError):
            print(f"[JUmPER]: Invalid cell range format: {cell_str}")
        return None

    @line_magic
    def perfmonitor_plot(self, line):
        """Open interactive plot with widgets for exploring performance data"""
        self._skip_report = True
        if not self.monitor:
            return print("[JUmPER]: No active performance monitoring session")
        self.visualizer.plot()

    @line_magic
    def perfmonitor_enable_perfreports(self, line):
        """Enable automatic performance reports after each cell execution"""
        self._skip_report = True
        args = self._parse_arguments(line)
        if args is None:
            return
        self.perfreports_level = args.level
        self.print_perfreports = True
        msg = (
            f"[JUmPER]: Performance reports enabled for each cell "
            f"(level: {self.perfreports_level})"
        )
        print(msg)

    @line_magic
    def perfmonitor_disable_perfreports(self, line):
        """Disable automatic performance reports after cell execution"""
        self._skip_report = True
        self.print_perfreports = False
        print("[JUmPER]: Performance reports disabled")

    @line_magic
    def perfmonitor_perfreport(self, line):
        """Show performance report with optional cell range and level
        filters"""
        self._skip_report = True
        if not self.reporter:
            return print("[JUmPER]: No active performance monitoring session")
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
            return print("[JUmPER]: No active performance monitoring session")
        parts = line.strip().split()
        filename = "performance_data.csv"
        if parts and not parts[0].startswith("--"):
            filename = parts[0]
            line = " ".join(parts[1:])
        args = self._parse_arguments(line)
        if not args:
            usage_msg = (
                "[JUmPER]: Usage: %perfmonitor_export_perfdata "
                "[filename] --level LEVEL"
            )
            return print(usage_msg)
        self.monitor.data.export(filename, level=args.level)
        print(
            f"[JUmPER]: Performance data ({args.level} level) exported to "
            f"{filename}"
        )

    @line_magic
    def perfmonitor_perfdata_to_dataframe(self, line):
        """Export performance data to dataframe with specified monitoring
        level"""
        self._skip_report = True
        if not self.monitor:
            return print("[JUmPER]: No active performance monitoring session")
        parts = line.strip().split()

        dataframe_name = None
        if parts and not parts[0].startswith("--"):
            dataframe_name = parts[0]
            line = " ".join(parts[1:])
        args = self._parse_arguments(line)
        if not args:
            usage_msg = (
                "[JUmPER]: Usage: %perfmonitor_perfdata_to_dataframe "
                "[df_name] --level LEVEL"
            )
            return print(usage_msg)
        dataframe_value = self.monitor.data.view(level=args.level)
        self.shell.push({dataframe_name: dataframe_value})
        print(
            f"[JUmPER]: Performance data ({args.level} level) exported to "
            f"{dataframe_name}"
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
            "perfmonitor_start [interval] -- start monitoring "
            "(default: 1 second)",
            "perfmonitor_stop -- stop monitoring",
            "perfmonitor_perfreport [--cell RANGE] [--level LEVEL] -- "
            "show report",
            "perfmonitor_plot -- interactive plot with widgets for data "
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
        print("[JUmPER]: Available commands:")
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
    print("[JUmPER]: Perfmonitor extension loaded.")


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
