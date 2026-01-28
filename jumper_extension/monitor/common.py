import logging
import os
import threading
import time
from typing import Dict, Optional, Protocol, runtime_checkable

import pandas as pd
import psutil

from jumper_extension.adapters.data import PerformanceData
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
from jumper_extension.utilities import detect_memory_limit, get_available_levels

logger = logging.getLogger("extension")

@runtime_checkable
class MonitorProtocol(Protocol):
    # required readable attributes
    interval: float
    data: "PerformanceData"
    start_time: Optional[float]
    num_cpus: int
    num_system_cpus: int
    num_gpus: int
    gpu_memory: float
    memory_limits: dict
    cpu_handles: list[int]
    gpu_name: str
    # session state
    is_imported: bool
    session_source: Optional[str]

    # required control & lifecycle
    running: bool
    def start(self, interval: float = 1.0) -> None: ...
    def stop(self) -> None: ...


class PerformanceMonitor:
    def __init__(self):
        self.interval = 1.0
        self.running = False
        self.start_time = None
        self.stop_time = None
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
        self.num_gpus = len(self.nvidia_gpu_handles) + len(
            self.amd_gpu_handles
        )
        self.metrics = [
            "cpu",
            "memory",
            "io_read",
            "io_write",
            "io_read_count",
            "io_write_count",
        ]

        if self.num_gpus:
            self.metrics.extend(["gpu_util", "gpu_band", "gpu_mem"])

        self.data = PerformanceData(
            self.num_cpus, self.num_system_cpus, self.num_gpus
        )
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
        return self._process_backend.get_filtered_processes(
            level, mode, handle
        )

    def _safe_proc_call(self, proc, proc_func, default=0):
        return self._process_backend.safe_proc_call(proc, proc_func, default)

    def _collect_cpu(self, level="process"):
        return self._cpu_backend.collect(level)

    def _collect_memory(self, level="process"):
        return self._memory_backend.collect(level)

    def _collect_io(self, level="process"):
        return self._io_backend.collect(level)

    def _collect_gpu(self, level="process"):
        if self.num_gpus == 0:
            return [], [], []

        self._validate_level(level)
        gpu_util, gpu_band, gpu_mem = [], [], []

        for backend in self._gpu_backends:
            b_util, b_band, b_mem = backend.collect(level)
            gpu_util.extend(b_util)
            gpu_band.extend(b_band)
            gpu_mem.extend(b_mem)

        return gpu_util, gpu_band, gpu_mem


    def _collect_metrics(self):
        time_mark = time.perf_counter()
        return tuple(
            (
                time_mark,
                self._collect_cpu(level),
                self._collect_memory(level),
                *self._collect_gpu(level),
                self._collect_io(level),
            )
            for level in self.levels
        )

    def _collect_data(self):
        while self.running:
            time_start_measurement = time.perf_counter()
            self.process_pids = self._get_process_pids()
            metrics = self._collect_metrics()
            for level, data_tuple in zip(self.levels, metrics):
                self.data.add_sample(level, *data_tuple)
            time_measurement = time.perf_counter() - time_start_measurement
            self.n_measurements += 1
            if time_measurement > self.interval:
                """
                logger.warning(
                    EXTENSION_INFO_MESSAGES[
                        ExtensionInfoCode.IMPRECISE_INTERVAL
                    ].format(interval=self.interval),
                    end="\r",
                )
                """
                self.n_missed_measurements += 1
            else:
                time.sleep(self.interval - time_measurement)

    def start(self, interval: float = 1.0):
        if self.running:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.MONITOR_ALREADY_RUNNING
                ]
            )
            return
        self.interval = interval
        self.start_time = time.perf_counter()
        self.running = True
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
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        self.stop_time = time.perf_counter()
        logger.info(
            EXTENSION_INFO_MESSAGES[ExtensionInfoCode.MONITOR_STOPPED].format(
                seconds=self.stop_time - self.start_time
            )
        )
        logger.info(
            EXTENSION_INFO_MESSAGES[ExtensionInfoCode.MISSED_MEASUREMENTS].format(
                perc_missed_measurements=self.n_missed_measurements / self.n_measurements
            )
        )


class MonitorUnavailableError(RuntimeError):
    """This monitor is a stub and cannot be used."""


class UnavailablePerformanceMonitor:
    """
    A stub that type-checks against PerformanceMonitor Protocol but fails at runtime.

    - Declares all required attributes for structural typing.
    - Any attribute access or method call raises MonitorUnavailableError,
      except 'running', which is always readable and returns False.
    """

    # --- Protocol surface ---
    interval: float
    data: "PerformanceData"
    start_time: Optional[float]
    num_cpus: int
    num_system_cpus: int
    num_gpus: int
    gpu_memory: float
    memory_limits: dict
    cpu_handles: list[int]
    gpu_name: str
    running: bool

    def start(self, interval: float = 1.0) -> None: ...
    def stop(self) -> None: ...

    # --- Runtime behavior ---
    def __init__(self, reason: str = "Performance monitor is not available"):
        object.__setattr__(self, "_reason", reason)

    def __getattribute__(self, name: str):
        # allow a few safe attributes + running
        if name in {
            "_reason", "__class__", "__repr__", "__str__",
            "__init__", "__getattribute__", "__setattr__",
            "__dict__", "__annotations__"
        }:
            return object.__getattribute__(self, name)

        if name == "running":
            return False

        reason = object.__getattribute__(self, "_reason")
        raise MonitorUnavailableError(f"Access to '{name}' is not allowed: {reason}")

    def __setattr__(self, name: str, value):
        if name in {"_reason", "__dict__", "__annotations__"}:
            return object.__setattr__(self, name, value)
        reason = object.__getattribute__(self, "_reason")
        raise MonitorUnavailableError(f"Setting '{name}' is not allowed: {reason}")

    def __repr__(self) -> str:
        return f"<UnavailablePerformanceMonitor: {self._reason}>"


class OfflinePerformanceMonitor:
    """Offline monitor that satisfies MonitorProtocol.

    It holds static data frames plus metadata from a manifest; does not collect live data.
    """

    def __init__(
        self,
        *,
        manifest: Dict,
        perf_dfs: Dict[str, pd.DataFrame],
        source: Optional[str] = None,
    ):
        monitor_info = manifest.get("monitor", {})

        # Protocol surface
        self.interval = float(monitor_info.get("interval", 1.0) or 1.0)
        self.running = False
        self.start_time = monitor_info.get("start_time")
        self.stop_time = monitor_info.get("stop_time")

        # Hardware/context
        self.num_cpus = int(monitor_info.get("num_cpus", 0) or 0)
        self.num_system_cpus = int(monitor_info.get("num_system_cpus", self.num_cpus) or self.num_cpus)
        self.num_gpus = int(monitor_info.get("num_gpus", 0) or 0)
        self.gpu_memory = float(monitor_info.get("gpu_memory", 0.0) or 0.0)
        self.gpu_name = monitor_info.get("gpu_name", "") or ""
        self.cpu_handles = monitor_info.get("cpu_handles", []) or []
        self.memory_limits = monitor_info.get("memory_limits", {}) or {}

        # Performance data container
        self.data = PerformanceData(
            self.num_cpus,
            self.num_system_cpus,
            self.num_gpus,
        )
        for level, df in (perf_dfs or {}).items():
            try:
                self.data._validate_level(level)
            except Exception:
                pass
            self.data.data[level] = df

        # Imported session state
        self.is_imported = True
        self.session_source = source

    # No-op lifecycle
    def start(self, interval: float = 1.0) -> None:
        self.interval = interval
        self.running = False

    def stop(self) -> None:
        self.running = False
