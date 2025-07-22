import pandas as pd


class PerformanceData:
    def __init__(self, num_cpus, num_gpus):
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.data = self._initialize_dataframe()

    def _initialize_dataframe(self):
        columns = [
            "time",
            "memory",
            "io_read_count",
            "io_write_count",
            "io_read",
            "io_write",
            "cpu_util_avg",
            "cpu_util_min",
            "cpu_util_max",
        ]

        # Add individual CPU columns
        for i in range(self.num_cpus):
            columns.append(f"cpu_util_{i}")

        if self.num_gpus > 0:
            columns.extend(
                [
                    "gpu_util_avg",
                    "gpu_util_min",
                    "gpu_util_max",
                    "gpu_band_avg",
                    "gpu_band_min",
                    "gpu_band_max",
                    "gpu_mem_avg",
                    "gpu_mem_min",
                    "gpu_mem_max",
                ]
            )

            # Add individual GPU columns
            for i in range(self.num_gpus):
                columns.extend([f"gpu_util_{i}", f"gpu_band_{i}", f"gpu_mem_{i}"])

        return pd.DataFrame(columns=columns)

    def view(self, slice_=None):
        if slice_ is None:
            return self.data
        return self.data.iloc[slice_[0] : slice_[1] + 1]

    def add_sample(
        self,
        time_mark,
        cpu_util_per_core,
        memory,
        gpu_util,
        gpu_band,
        gpu_mem,
        io_counters,
    ):
        row_data = {
            "time": time_mark,
            "memory": memory,
            "io_read_count": io_counters[0],
            "io_write_count": io_counters[1],
            "io_read": io_counters[2],
            "io_write": io_counters[3],
            "cpu_util_avg": sum(cpu_util_per_core) / self.num_cpus,
            "cpu_util_min": min(cpu_util_per_core),
            "cpu_util_max": max(cpu_util_per_core),
        }

        # Add individual CPU utilization
        for i in range(self.num_cpus):
            row_data[f"cpu_util_{i}"] = cpu_util_per_core[i]

        if self.num_gpus > 0:
            row_data.update(
                {
                    "gpu_util_avg": sum(gpu_util) / self.num_gpus,
                    "gpu_util_min": min(gpu_util),
                    "gpu_util_max": max(gpu_util),
                    "gpu_band_avg": sum(gpu_band) / self.num_gpus,
                    "gpu_band_min": min(gpu_band),
                    "gpu_band_max": max(gpu_band),
                    "gpu_mem_avg": sum(gpu_mem) / self.num_gpus,
                    "gpu_mem_min": min(gpu_mem),
                    "gpu_mem_max": max(gpu_mem),
                }
            )

            # Add individual GPU metrics
            for i in range(self.num_gpus):
                row_data[f"gpu_util_{i}"] = gpu_util[i]
                row_data[f"gpu_band_{i}"] = gpu_band[i]
                row_data[f"gpu_mem_{i}"] = gpu_mem[i]

        self.data.loc[len(self.data)] = row_data

    def export(self, filename="performance_data.csv"):
        self.data.to_csv(filename, index=False)
