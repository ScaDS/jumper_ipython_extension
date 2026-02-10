import pandas as pd
import logging
import logging.config

from .utilities import get_available_levels

from .extension_messages import (
    ExtensionErrorCode,
    ExtensionInfoCode,
    EXTENSION_ERROR_MESSAGES,
    EXTENSION_INFO_MESSAGES,
)

logger = logging.getLogger("extension")


class PerformanceData:
    def __init__(self, num_cpus, num_system_cpus, num_gpus):
        self.num_cpus = num_cpus
        self.num_system_cpus = num_system_cpus
        self.num_gpus = num_gpus
        self.levels = get_available_levels()
        self.data = {
            level: self._initialize_dataframe(level) for level in self.levels
        }

    def _validate_level(self, level):
        if level not in self.levels:
            raise ValueError(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.INVALID_LEVEL
                ].format(level=level, levels=self.levels)
            )

    def _initialize_dataframe(self, level):

        effective_num_cpus = (
            self.num_system_cpus if level == "system" else self.num_cpus
        )

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
        ] + [f"cpu_util_{i}" for i in range(effective_num_cpus)]

        if self.num_gpus > 0:
            gpu_metrics = ["util", "band", "mem", "power"]
            columns.extend(
                [
                    f"gpu_{metric}_{stat}"
                    for metric in gpu_metrics
                    for stat in ["avg", "min", "max"]
                ]
            )
            columns.extend(
                [
                    f"gpu_{metric}_{i}"
                    for i in range(self.num_gpus)
                    for metric in gpu_metrics
                ]
            )

        return pd.DataFrame(columns=columns)

    def view(self, level="process", slice_=None):
        """View data for a specific level with optional slicing."""
        self._validate_level(level)
        return (
            self.data[level]
            if slice_ is None
            else self.data[level].iloc[slice_[0] : slice_[1] + 1]
        )

    def add_sample(
        self,
        level,
        time_mark,
        cpu_util_per_core,
        memory,
        gpu_util,
        gpu_band,
        gpu_mem,
        gpu_power,
        io_counters,
    ):
        self._validate_level(level)
        effective_num_cpus = (
            self.num_system_cpus if level == "system" else self.num_cpus
        )

        last_timestamp = 0
        if len(self.data[level]):
            last_timestamp = self.data[level].loc[len(self.data[level]) - 1][
                "time"
            ]

        cumulative_metrics_ratio = time_mark - last_timestamp
        row_data = {
            "time": time_mark,
            "memory": memory,
            "io_read_count": io_counters[0] / cumulative_metrics_ratio,
            "io_write_count": io_counters[1] / cumulative_metrics_ratio,
            "io_read": io_counters[2] / cumulative_metrics_ratio,
            "io_write": io_counters[3] / cumulative_metrics_ratio,
            "cpu_util_avg": sum(cpu_util_per_core) / effective_num_cpus,
            "cpu_util_min": min(cpu_util_per_core),
            "cpu_util_max": max(cpu_util_per_core),
            **{
                f"cpu_util_{i}": cpu_util_per_core[i]
                for i in range(effective_num_cpus)
            },
        }

        if self.num_gpus > 0:
            gpu_data = {"util": gpu_util, "band": gpu_band, "mem": gpu_mem, "power": gpu_power}
            for metric, values in gpu_data.items():
                row_data.update(
                    {
                        f"gpu_{metric}_avg": sum(values) / self.num_gpus,
                        f"gpu_{metric}_min": min(values),
                        f"gpu_{metric}_max": max(values),
                        **{
                            f"gpu_{metric}_{i}": values[i]
                            for i in range(self.num_gpus)
                        },
                    }
                )

        self.data[level].loc[len(self.data[level])] = row_data

    def export(self, filename="performance_data.csv", level="process"):
        """Export performance data to CSV."""
        self._validate_level(level)
        if len(self.data[level]) == 0:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.NO_PERFORMANCE_DATA
                ]
            )
            return

        self.data[level].to_csv(filename, index=False)

        logger.info(
            EXTENSION_INFO_MESSAGES[ExtensionInfoCode.EXPORT_SUCCESS].format(
                filename=filename
            )
        )
