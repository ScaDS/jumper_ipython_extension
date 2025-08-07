from .utilities import filter_perfdata


class PerformanceReporter:
    def __init__(self, monitor, cell_history, min_duration=None):
        self.monitor = monitor
        self.cell_history = cell_history
        self.min_duration = min_duration

    def print(self, cell_range=None, level="process"):
        """Print performance report"""
        if not self.monitor:
            print("[JUmPER]: No active performance monitoring session")
            return

        if cell_range is None:
            valid_cells = self.cell_history.view()

            if len(valid_cells) > 0:
                # Filter for non-short cells
                min_duration = self.min_duration if self.min_duration is not None else 0
                non_short_cells = valid_cells[valid_cells["duration"] >= min_duration]

                if len(non_short_cells) > 0:
                    # Get the last non-short cell index
                    last_valid_cell_idx = int(non_short_cells.iloc[-1]["index"])
                    cell_range = (last_valid_cell_idx, last_valid_cell_idx)
                else:
                    print("[JUmPER]: No non-short cells found")
                    return
            else:
                return

        # Filter cell history data first using cell_range
        start_idx, end_idx = cell_range
        filtered_cells = self.cell_history.view(start_idx, end_idx + 1)

        perfdata = self.monitor.data.view(level=level)
        perfdata = filter_perfdata(filtered_cells, perfdata, compress_idle=False)

        # Check if non-empty, otherwise print results
        if perfdata.empty:
            print("[JUmPER]: No performance data available")
            return

        # Calculate total duration of selected cells
        total_duration = filtered_cells["duration"].sum()

        print("-" * 40)
        print("JUmPER Performance Report")
        print("-" * 40)
        print(
            f"Duration: {total_duration:.2f}s ({len(filtered_cells)} cell{'s' if len(filtered_cells) != 1 else ''})"
        )
        print("-" * 40)

        # Report table
        metrics = [
            (f"CPU Util (Across {self.monitor.num_cpus} CPUs)", "cpu_util_avg", "-"),
            ("Memory (GB)", "memory", f"{self.monitor.memory:.2f}"),
            (f"GPU Util (Across {self.monitor.num_gpus} GPUs)", "gpu_util_avg", "-"),
            ("GPU Memory (GB)", "gpu_mem_avg", f"{self.monitor.gpu_memory:.2f}"),
        ]

        print(f"{'Metric':<25} {'AVG':<8} {'MIN':<8} {'MAX':<8} {'TOTAL':<8}")
        print("-" * 65)
        for name, col, total in metrics:
            if col in perfdata.columns:
                print(
                    f"{name:<25} {perfdata[col].mean():<8.2f} "
                    f"{perfdata[col].min():<8.2f} {perfdata[col].max():<8.2f} "
                    f"{total:<8}"
                )
