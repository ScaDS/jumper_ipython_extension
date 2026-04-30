"""StorageHandler layer — transforms raw collector output into flat dicts.

Each handler converts one specific raw type into a ``dict[str, float]``
ready to be appended as a DataFrame row.  The four concrete handlers cover
all current collector return types without requiring those collectors to be
modified.
"""

from __future__ import annotations

import time as _time
from typing import Protocol


class StorageHandler(Protocol):
    def transform(self, raw, level: str) -> dict[str, float]:
        """Return a flat column→value dict for one sample."""
        ...


class ScalarHandler:
    """raw: float  →  {column: value}"""

    def __init__(self, column: str) -> None:
        self._column = column

    def transform(self, raw, level: str) -> dict[str, float]:
        return {self._column: float(raw)}


class PerDeviceAggregateHandler:
    """raw: list[float]  →  {prefix_0: v, …, prefix_avg, prefix_min, prefix_max}"""

    def __init__(self, prefix: str) -> None:
        self._prefix = prefix

    def transform(self, raw: list, level: str) -> dict[str, float]:
        if not raw:
            return {}
        n = len(raw)
        result: dict[str, float] = {
            f"{self._prefix}{i}": float(raw[i]) for i in range(n)
        }
        result[f"{self._prefix}avg"] = sum(raw) / n
        result[f"{self._prefix}min"] = float(min(raw))
        result[f"{self._prefix}max"] = float(max(raw))
        return result


class PerDeviceMultiAggregateHandler:
    """raw: tuple[list[float], …]  →  per-metric fan-out via PerDeviceAggregate.

    Example with prefix="gpu_", metrics=["util","band","mem"]:
      gpu_util_0, gpu_util_avg, …, gpu_band_0, …, gpu_mem_0, …
    """

    def __init__(self, prefix: str, metrics: list) -> None:
        self._prefix = prefix
        self._metrics = metrics

    def transform(self, raw: tuple, level: str) -> dict[str, float]:
        result: dict[str, float] = {}
        for metric, values in zip(self._metrics, raw):
            sub = PerDeviceAggregateHandler(f"{self._prefix}{metric}_")
            result.update(sub.transform(list(values), level))
        return result


class CumulativeRateHandler:
    """raw: list[int]  →  per-second delta rates, one column per counter.

    Stateful per level: tracks previous counter values and the previous
    timestamp so it can compute (delta / dt) for each counter.
    Returns zeros on the first call (no previous state available).
    """

    def __init__(self, columns: list) -> None:
        self._columns = columns
        self._last_counters: dict[str, list] = {}
        self._last_time: dict[str, float] = {}

    def transform(self, raw: list, level: str) -> dict[str, float]:
        now = _time.perf_counter()
        last = self._last_counters.get(level)
        last_t = self._last_time.get(level, now)
        dt = max(now - last_t, 1e-6)

        self._last_counters[level] = list(raw)
        self._last_time[level] = now

        if last is None:
            return {col: 0.0 for col in self._columns}

        return {
            col: max(0.0, float(raw[i] - last[i])) / dt
            for i, col in enumerate(self._columns)
        }


def make_handler(storage_cfg: dict) -> StorageHandler:
    """Instantiate the right StorageHandler from a storage-config dict."""
    t = storage_cfg["type"]
    if t == "scalar":
        return ScalarHandler(column=storage_cfg["column"])
    if t == "per_device_aggregate":
        return PerDeviceAggregateHandler(prefix=storage_cfg["prefix"])
    if t == "per_device_multi_aggregate":
        return PerDeviceMultiAggregateHandler(
            prefix=storage_cfg["prefix"],
            metrics=storage_cfg["metrics"],
        )
    if t == "cumulative_rate":
        return CumulativeRateHandler(columns=storage_cfg["columns"])
    raise ValueError(f"Unknown storage type: {t!r}")
