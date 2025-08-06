from IPython.core.magic import Magics, line_magic, magics_class
import argparse
import shlex

from .cell_history import CellHistory
from .monitor import PerformanceMonitor
from .reporter import PerformanceReporter
from .visualizer import PerformanceVisualizer

_perfmonitor_magics = None


@magics_class
class perfmonitorMagics(Magics):
    def __init__(self, shell):
        super().__init__(shell)
        self.monitor = None
        self.visualizer = None
        self.reporter = None
        self.cell_history = CellHistory()
        self.print_perfreports = False
        self._skip_report = False
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
            self.reporter.print(cell_range=None)
        self._skip_report = False

    @line_magic
    def perfmonitor_resources(self, line):
        """Show available hardware"""
        self._skip_report = True
        if not self.monitor:
            print("[JUmPER]: No active performance monitoring session")
            return
        print("[JUmPER]:")
        print(f"  CPUs: {self.monitor.num_cpus}")
        print(f"    CPU affinity: {self.monitor.cpu_handles}")
        print(f"  Memory: {self.monitor.memory} GB")
        print(f"  GPUs: {self.monitor.num_gpus}")
        if self.monitor.num_gpus:
            print(f"    {self.monitor.gpu_name}, {self.monitor.gpu_memory} GB")

    @line_magic
    def cell_history(self, line):
        self._skip_report = True
        self.cell_history.show_itable()

    @line_magic
    def perfmonitor_start(self, line):
        self._skip_report = True
        if self.monitor and self.monitor.running:
            print("[JUmPER]: Performance monitoring already running")
            return

        interval = 1.0
        if line:
            try:
                interval = float(line)
            except ValueError:
                print(f"[JUmPER]: Invalid interval value: {line}")
                return

        self.monitor = PerformanceMonitor(interval=interval)
        self.monitor.start()
        self.visualizer = PerformanceVisualizer(self.monitor, self.cell_history, min_duration=interval)
        self.reporter = PerformanceReporter(self.monitor, self.cell_history, min_duration=interval)
        self.min_duration = interval

    @line_magic
    def perfmonitor_stop(self, line):
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

        default_metrics = ["cpu", "cpu_all", "mem", "io"]
        if self.monitor.num_gpus:
            default_metrics.extend(["gpu", "gpu_all"])

        parser.add_argument(
            "--metrics",
            nargs="+",
            default=default_metrics,
            help="Metric subsets",
        )
        parser.add_argument(
            "--show-idle", action="store_true", help="Show idle periods (do not compress time axis)"
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
                start_idx = 0 if start_str == "" else int(start_str)
                end_idx = max_idx if end_str == "" else int(end_str)
            else:
                start_idx = end_idx = int(cell_str)

            if not (0 <= start_idx <= end_idx <= max_idx):
                print(f"[JUmPER]: Invalid cell range: {cell_str} (valid range: 0-{max_idx})")
                return None

            return (start_idx, end_idx)

        except (ValueError, IndexError):
            print(f"[JUmPER]: Invalid cell range format: {cell_str}")
            return None

    @line_magic
    def perfmonitor_plot(self, line):
        self._skip_report = True
        if not self.monitor:
            print("[JUmPER]: No active performance monitoring session")
            return

        args = self._parse_arguments(line)
        if args is None:
            return

        cell_range = None
        if args.cell is not None:
            cell_range = self._parse_cell_range(args.cell, self.cell_history)
            if cell_range is None:
                return

        self.visualizer.plot(
            metric_subsets=args.metrics,
            cell_range=cell_range,
            show_idle=args.show_idle,
        )

    @line_magic
    def perfmonitor_enable_perfreports(self, line):
        self._skip_report = True
        self.print_perfreports = True
        print("[JUmPER]: Performance reports enabled for each cell")

    @line_magic
    def perfmonitor_disable_perfreports(self, line):
        self._skip_report = True
        self.print_perfreports = False
        print("[JUmPER]: Performance reports disabled")

    @line_magic
    def perfmonitor_perfreport(self, line):
        self._skip_report = True

        if not self.reporter:
            print("[JUmPER]: No active performance monitoring session")
            return

        args = self._parse_arguments(line)
        if args is None:
            return

        cell_range = None
        if args.cell is not None:
            cell_range = self._parse_cell_range(args.cell, self.cell_history)
            if cell_range is None:
                return

        self.reporter.print(cell_range=cell_range)

    @line_magic
    def perfmonitor_export_perfdata(self, line):
        """Export performance data"""
        self._skip_report = True
        if not self.monitor:
            print("[JUmPER]: No active performance monitoring session")
            return
        filename = line.strip() or "performance_data.csv"
        self.monitor.data.export(filename)
        print(f"[JUmPER]: Performance data exported to {filename}")

    @line_magic
    def perfmonitor_export_cell_history(self, line):
        """Export cell history to JSON or CSV format"""
        self._skip_report = True
        filename = line.strip() or "cell_history.json"
        self.cell_history.export(filename)

    @line_magic
    def perfmonitor_help(self, line):
        """Show help information"""
        self._skip_report = True
        commands = [
            "perfmonitor_help -- show this help",
            "perfmonitor_resources -- show available hardware",
            "cell_history [--df] -- show cell execution history (or DataFrame)",
            "cell_history_stats -- show cell execution statistics",
            "perfmonitor_start [seconds] -- start monitoring",
            "perfmonitor_stop -- stop monitoring",
            "perfmonitor_perfreport --cell RANGE -- show performance report",
            "perfmonitor_plot [--cell RANGE] [--metrics SUBSET1 SUBSET2 ...] -- plot",
            "perfmonitor_enable_perfreports -- enable auto-reports",
            "perfmonitor_disable_perfreports -- disable auto-reports",
            "perfmonitor_export_perfdata [filename] -- export data to CSV",
            "perfmonitor_export_cell_history [filename] -- export history to JSON/CSV",
        ]
        print("[JUmPER]: Available commands:")
        for cmd in commands:
            print(f"  %{cmd}")

        print(
            "\nMetric subsets: cpu_all, gpu_all, cpu, gpu, mem, io "
            "(default: cpu, gpu, mem, io)"
        )
        print("Cell ranges: 5 (single), 2:8 (range), :5 (from start)")
        print("Options: --show-idle (show idle periods, do not compress time axis)")

        print("\nExamples:")
        print("  %perfmonitor_plot --cell 5 --metrics cpu mem")
        print("  %perfmonitor_plot --show-idle")
        print("  %perfmonitor_perfreport --cell 2:8")


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
        ipython.events.unregister("pre_run_cell", _perfmonitor_magics.pre_run_cell)
        ipython.events.unregister("post_run_cell", _perfmonitor_magics.post_run_cell)
        if _perfmonitor_magics.monitor:
            _perfmonitor_magics.monitor.stop()
        _perfmonitor_magics = None
