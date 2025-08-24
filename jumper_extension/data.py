import json
import os
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
            gpu_metrics = ["util", "band", "mem"]
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

    def _attach_cell_index(self, df, cell_history):
        result = df.copy()
        result["cell_index"] = pd.NA
        times = result["time"].to_numpy()
        for row in cell_history.data.itertuples(index=False):
            mask = (times >= row.start_time) & (times <= row.end_time)
            result.loc[mask, "cell_index"] = row.cell_index
        return result

    def view(self, level="process", slice_=None, cell_history=None):
        """View data for a specific level with optional slicing."""
        self._validate_level(level)
        base = (
            self.data[level]
            if slice_ is None
            else self.data[level].iloc[slice_[0] : slice_[1] + 1]
        )
        return (
            self._attach_cell_index(base, cell_history)
            if cell_history is not None
            else base
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
            gpu_data = {"util": gpu_util, "band": gpu_band, "mem": gpu_mem}
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

    def export(
        self,
        filename="performance_data.csv",
        level="process",
        cell_history=None,
    ):
        """Export performance data to JSON or CSV."""
        self._validate_level(level)
        if len(self.data[level]) == 0:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.NO_PERFORMANCE_DATA
                ]
            )
            return

        # Determine format from filename extension
        _, ext = os.path.splitext(filename)
        format = ext.lower().lstrip(".")
        
        # Default to csv if no extension provided
        if not format:
            format = "csv"
            filename += ".csv"

        df_to_write = (
            self._attach_cell_index(self.data[level], cell_history)
            if cell_history is not None
            else self.data[level]
        )

        if format == "json":
            with open(filename, "w") as f:
                json.dump(df_to_write.to_dict("records"), f, indent=2)
        elif format == "csv":
            df_to_write.to_csv(filename, index=False)
        else:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.UNSUPPORTED_EXPORT_FORMAT
                ].format(
                    format=format,
                    supported_formats=", ".join(["json", "csv"]),
                )
            )
            return

        logger.info(
            EXTENSION_INFO_MESSAGES[ExtensionInfoCode.EXPORT_SUCCESS].format(
                filename=filename
            )
        )
