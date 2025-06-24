from IPython.core.magic import Magics, magics_class, line_magic
from .performance_monitor import PerformanceMonitor
from .performance_visualizer import PerformanceVisualizer
from .cell_history import CellHistory

_perfmonitor_magics = None

@magics_class
class perfmonitorMagics(Magics):
    def __init__(self, shell):
        super().__init__(shell)
        self.monitor = None
        self.visualizer = None
        self.cell_history = CellHistory()
        self.print_perfreports = False

    def pre_run_cell(self, info):
        self.cell_history.start_cell(info.raw_cell)
    
    def post_run_cell(self, result):
        self.cell_history.end_cell(result.result)
        if self.monitor and self.print_perfreports:
            self._perfreport()

    def _parse_cell_number(self, line):
        """Parse cell number from command line"""
        if not line:
            return None
        try:
            cell_num = int(line)
            return self.cell_history.cell_timestamps[cell_num]
        except (ValueError, IndexError):
            print(f"Invalid cell index: {line}")
            return False

    @line_magic
    def perfmonitor_resources(self, line):
        """Show available hardware"""
        if not self.monitor:
            print("No active performance monitoring session")
            return
        print(f"  CPUs: {self.monitor.num_cpus}")
        print(f"    CPU affinity: {self.monitor.cpu_handles}")
        print(f"  Memory: {self.monitor.memory} GB")
        print(f"  GPUs: {self.monitor.num_gpus}")
        if self.monitor.num_gpus:
            print(f"    {self.monitor.gpu_name}, {self.monitor.gpu_memory} GB")

    @line_magic
    def cell_history(self, line):
        self.cell_history.print()

    @line_magic
    def perfmonitor_start(self, line):
        if self.monitor and self.monitor.running:
            print("Performance monitoring already running")
            return
        
        interval = 1.0
        if line:
            try:
                interval = float(line)
            except ValueError:
                print(f"Invalid interval value: {line}")
                return
        
        self.monitor = PerformanceMonitor(interval=interval)
        self.monitor.start()
        self.visualizer = PerformanceVisualizer(
            self.monitor.cpu_handles, self.monitor.memory, self.monitor.gpu_memory)

    @line_magic
    def perfmonitor_stop(self, line):
        if not self.monitor:
            print("No active performance monitoring session")
            return
        self.monitor.stop()

    @line_magic
    def perfmonitor_plot(self, line):
        if not self.monitor:
            print("No active performance monitoring session")
            return
        
        cell_marks = self._parse_cell_number(line)
        if cell_marks is False:  # Error parsing
            return
        
        df = self.monitor.data.to_dataframe()
        if cell_marks:
            start_mark, end_mark = cell_marks
            df = df[(df['time'] >= start_mark) & (df['time'] <= end_mark)]
        df['time'] -= self.monitor.start_time

        if df.empty:
            print("No performance data available")
            return

        self.visualizer.plot(df, self.monitor.metrics)

    def _perfreport(self, cell_marks=None):
        """Generate performance report"""
        if not self.monitor:
            print("No active performance monitoring session")
            return

        print("-" * 40)
        print("Performance Report")
        print("-" * 40)

        if not cell_marks:
            cell_marks = self.cell_history.cell_timestamps[-1]
        
        start_mark, end_mark = cell_marks
        duration = end_mark - start_mark
        print(f"Duration: {duration:.2f}s")

        df = self.monitor.data.to_dataframe()
        df = df[(df['time'] >= start_mark) & (df['time'] <= end_mark)]
        if df.empty:
            print("No performance data available")
            return

        # Report table
        metrics = [
            ("CPU Util (Across CPUs)", "cpu_util_avg", "-"),
            ("Memory (GB)", "memory_usage_gb", f"{self.monitor.memory:.2f}"),
            ("GPU Util (Across GPUs)", "gpu_util_avg", "-"),
            ("GPU Memory (GB)", "gpu_mem_avg", f"{self.monitor.gpu_memory:.2f}")
        ]
        
        print(f"{'Metric':<25} {'AVG':<8} {'MIN':<8} {'MAX':<8} {'TOTAL':<8}")
        print("-" * 65)
        for name, col, total in metrics:
            if col in df.columns:
                print(f"{name:<25} {df[col].mean():<8.2f} {df[col].min():<8.2f} {df[col].max():<8.2f} {total:<8}")
            
    @line_magic
    def perfmonitor_enable_perfreports(self, line):
        self.print_perfreports = True
        print("Performance reports enabled for each cell")
    
    @line_magic
    def perfmonitor_disable_perfreports(self, line):
        self.print_perfreports = False
        print("Performance reports disabled")

    @line_magic
    def perfmonitor_perfreport(self, line):
        cell_marks = self._parse_cell_number(line)
        if cell_marks is not False:
            self._perfreport(cell_marks)

    @line_magic
    def perfmonitor_export_perfdata(self, line):
        """Export performance data"""
        if not self.monitor:
            print("No active performance monitoring session")
            return
        filename = line.strip() or 'performance_data.csv'
        self.monitor.data.export(filename)
        print(f"Performance data exported to {filename}")

    @line_magic
    def perfmonitor_export_cell_history(self, line):
        """Export cell history"""
        filename = line.strip() or 'cell_history.json'
        self.cell_history.export(filename)
        print(f"Cell history exported to {filename}")

    @line_magic
    def perfmonitor_help(self, line):
        """Show help information"""
        commands = [
            "perfmonitor_help -- show this help",
            "perfmonitor_resources -- show available hardware",
            "cell_history -- show cell execution history", 
            "perfmonitor_start [seconds] -- start monitoring",
            "perfmonitor_stop -- stop monitoring",
            "perfmonitor_perfreport [cell] -- show performance report",
            "perfmonitor_plot [cell] -- plot performance data",
            "perfmonitor_enable_perfreports -- enable auto-reports",
            "perfmonitor_disable_perfreports -- disable auto-reports",
            "perfmonitor_export_perfdata [filename] -- export data to CSV",
            "perfmonitor_export_cell_history [filename] -- export history to JSON"
        ]
        print("Available commands:")
        for cmd in commands:
            print(f"  %{cmd}")

def load_ipython_extension(ipython):
    global _perfmonitor_magics
    _perfmonitor_magics = perfmonitorMagics(ipython)
    ipython.events.register('pre_run_cell', _perfmonitor_magics.pre_run_cell)
    ipython.events.register('post_run_cell', _perfmonitor_magics.post_run_cell)
    ipython.register_magics(_perfmonitor_magics)
    print("Perfmonitor extension loaded.")

def unload_ipython_extension(ipython):
    global _perfmonitor_magics
    if _perfmonitor_magics:
        ipython.events.unregister('pre_run_cell', _perfmonitor_magics.pre_run_cell)
        ipython.events.unregister('post_run_cell', _perfmonitor_magics.post_run_cell)
        if _perfmonitor_magics.monitor:
            _perfmonitor_magics.monitor.stop()
        _perfmonitor_magics = None
