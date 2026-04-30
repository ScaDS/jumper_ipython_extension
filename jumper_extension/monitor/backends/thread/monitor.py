import logging
import os
import threading
import time
from typing import Optional

import psutil

from jumper_extension.adapters.data import NodeInfo, NodeDataStore
from jumper_extension.core.messages import (
    ExtensionErrorCode,
    ExtensionInfoCode,
    EXTENSION_ERROR_MESSAGES,
    EXTENSION_INFO_MESSAGES,
)
from jumper_extension.monitor.metrics.cpu.psutil import PsutilCpuBackend
from jumper_extension.monitor.metrics.gpu.common import GpuBackendDiscovery
from jumper_extension.monitor.metrics.io.psutil import PsutilIoBackend
from jumper_extension.monitor.metrics.memory.psutil import PsutilMemoryBackend
from jumper_extension.monitor.metrics.process.psutil import PsutilProcessBackend
from jumper_extension.monitor.metrics.storage import (
    ScalarHandler,
    PerDeviceAggregateHandler,
    PerDeviceMultiAggregateHandler,
    CumulativeRateHandler,
)
from jumper_extension.utilities import detect_memory_limit, get_available_levels

logger = logging.getLogger("extension")


class _CombinedGpuCollector:
    """Aggregates multiple GPU backends into a single collect() call."""

    def __init__(self, backends):
        self._backends = backends

    def collect(self, level: str):
        util, band, mem = [], [], []
        for backend in self._backends:
            backend_util, backend_band, backend_mem = backend.collect(level)
            util.extend(backend_util)
            band.extend(backend_band)
            mem.extend(backend_mem)
        return util, band, mem


class PerformanceMonitor:
    def __init__(self):
        self.interval = 1.0
        self.running = False
        self.start_time = None
        self.stop_time = None
        self.wallclock_start_time = None
        self.wallclock_stop_time = None
        self.monitor_thread = None
        self.process = psutil.Process()
        self.n_measurements = 0
        self.n_missed_measurements = 0
        """
        on MacOS cpu_affinity is not implemented in psutil
        (raises AttributeError)
        set the num_cpus to the number of cpus in the system
        same for cpu_affinity
        """
        try:
            self.cpu_handles = self.process.cpu_affinity()
            self.num_cpus = len(self.cpu_handles)
        except AttributeError:
            self.cpu_handles = []
            self.num_cpus = len(psutil.cpu_percent(percpu=True))
        self.num_system_cpus = len(psutil.cpu_percent(percpu=True))
        self.pid = os.getpid()
        self.uid = os.getuid()
        self.slurm_job = os.environ.get("SLURM_JOB_ID", 0)
        self.levels = get_available_levels()
        self.process_pids = []

        self.memory_limits = {
            level: detect_memory_limit(level, self.uid, self.slurm_job)
            for level in self.levels
        }

        self._process_backend = PsutilProcessBackend(self)
        self._cpu_backend = PsutilCpuBackend(self)
        self._memory_backend = PsutilMemoryBackend(self)
        self._io_backend = PsutilIoBackend(self)
        for backend in (
            self._process_backend,
            self._cpu_backend,
            self._memory_backend,
            self._io_backend,
        ):
            backend.setup()

        self.nvidia_gpu_handles = []
        self.amd_gpu_handles = []
        self.gpu_memory = 0
        self.gpu_name = ""
        self._gpu_backends = GpuBackendDiscovery(self).discover()
        for backend in self._gpu_backends:
            backend.setup()
        self.num_gpus = len(self.nvidia_gpu_handles) + len(self.amd_gpu_handles)

        self.metrics = [
            "cpu", "memory", "io_read", "io_write", "io_read_count", "io_write_count",
        ]
        if self.num_gpus:
            self.metrics.extend(["gpu_util", "gpu_band", "gpu_mem"])

        # Build the collector→handler pipeline
        _gpu_collector = _CombinedGpuCollector(self._gpu_backends)
        self._pipeline = [
            (self._cpu_backend,    PerDeviceAggregateHandler("cpu_util_")),
            (self._memory_backend, ScalarHandler("memory")),
            (_gpu_collector,       PerDeviceMultiAggregateHandler("gpu_", ["util", "band", "mem"])),
            (self._io_backend,     CumulativeRateHandler(
                ["io_read_count", "io_write_count", "io_read", "io_write"]
            )),
        ]

        node_info = NodeInfo(
            node="local",
            num_cpus=self.num_cpus,
            num_system_cpus=self.num_system_cpus,
            num_gpus=self.num_gpus,
            gpu_memory=self.gpu_memory,
            gpu_name=self.gpu_name,
            memory_limits=self.memory_limits,
            cpu_handles=self.cpu_handles,
        )
        self.nodes = NodeDataStore()
        self.nodes.register_node(node_info)

        # Bootstrap: warm up process snapshots and IO counter state,
        # then derive per-level column names for schema pre-population.
        self._process_backend.snapshot_metrics()
        columns_by_level: dict[str, list[str]] = {}
        for level in self.levels:
            row: dict = {"time": 0.0}
            for collector, handler in self._pipeline:
                try:
                    row.update(handler.transform(collector.collect(level), level))
                except Exception:
                    pass
            columns_by_level[level] = list(row.keys())
        self.nodes.init_node_schema("local", columns_by_level)

        # session state
        self.is_imported = False
        self.session_source = None

    def _get_process_pids(self):
        return self._process_backend.get_process_pids()

    def _validate_level(self, level):
        if level not in self.levels:
            raise ValueError(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.INVALID_LEVEL
                ].format(level=level, levels=self.levels)
            )

    def _filter_process(self, proc, mode):
        return self._process_backend.filter_process(proc, mode)

    def _get_filtered_processes(self, level="user", mode="cpu", handle=None):
        return self._process_backend.get_filtered_processes(level, mode, handle)

    def _safe_proc_call(self, proc, proc_func, default=0):
        return self._process_backend.safe_proc_call(proc, proc_func, default)

    def _collect_metrics(self):
        """Collect one sample per level; return a list of flat dicts."""
        self._process_backend.snapshot_metrics()
        time_mark = time.perf_counter()
        rows = []
        for level in self.levels:
            row: dict = {"time": time_mark}
            for collector, handler in self._pipeline:
                row.update(handler.transform(collector.collect(level), level))
            rows.append(row)
        return rows

    def _collect_data(self):
        """Collect metrics at a fixed cadence anchored to an absolute timeline.

        Uses ``threading.Event.wait`` instead of ``time.sleep`` so that:
        * each tick is scheduled relative to a fixed epoch, preventing
          per-iteration sleep-overshoot from accumulating into drift;
        * ``stop()`` can signal the event and wake the thread instantly
          instead of blocking up to one full interval on ``thread.join``.

        The GIL is released during ``Event.wait`` just like ``time.sleep``.
        """
        next_tick = time.perf_counter()
        while not self._stop_event.is_set():
            self.process_pids = self._get_process_pids()
            rows = self._collect_metrics()
            for level, row in zip(self.levels, rows):
                self.nodes.add_sample("local", level, row)
            self.n_measurements += 1

            next_tick += self.interval
            delay = next_tick - time.perf_counter()
            if delay > 0:
                self._stop_event.wait(delay)
            else:
                self.n_missed_measurements += 1
                next_tick = time.perf_counter()

    def start(self, interval: float = 1.0):
        if self.running:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.MONITOR_ALREADY_RUNNING]
            )
            return
        self.interval = interval
        self.start_time = time.perf_counter()
        self.wallclock_start_time = time.time()
        self.running = True
        self._stop_event = threading.Event()
        self.monitor_thread = threading.Thread(
            target=self._collect_data, daemon=True
        )
        self.monitor_thread.start()
        logger.info(
            EXTENSION_INFO_MESSAGES[ExtensionInfoCode.MONITOR_STARTED].format(
                pid=self.pid,
                interval=self.interval,
            )
        )

    def stop(self):
        self.running = False
        if hasattr(self, "_stop_event"):
            self._stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        self.stop_time = time.perf_counter()
        self.wallclock_stop_time = time.time()

        elapsed = self.stop_time - self.start_time
        expected = int(elapsed / self.interval) if self.interval > 0 else 0
        self.n_missed_measurements = max(0, expected - self.n_measurements)

        logger.info(
            EXTENSION_INFO_MESSAGES[ExtensionInfoCode.MONITOR_STOPPED].format(
                seconds=elapsed
            )
        )
        if self.n_measurements > 0:
            logger.info(
                EXTENSION_INFO_MESSAGES[ExtensionInfoCode.MISSED_MEASUREMENTS].format(
                    perc_missed_measurements=(
                        self.n_missed_measurements / expected
                        if expected > 0 else 0
                    )
                )
            )
