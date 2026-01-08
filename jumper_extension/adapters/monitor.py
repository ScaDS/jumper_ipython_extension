import logging
import os
import threading
import time
import unittest.mock
from typing import (
    Protocol,
    Optional,
    runtime_checkable,
    Dict,
    Iterable,
    Callable,
    Any,
)

import psutil

from jumper_extension.adapters.data import PerformanceData
import pandas as pd
from jumper_extension.core.messages import (
    ExtensionErrorCode,
    ExtensionInfoCode,
    EXTENSION_ERROR_MESSAGES,
    EXTENSION_INFO_MESSAGES,
)
from jumper_extension.utilities import (
    get_available_levels,
    is_slurm_available,
    detect_memory_limit,
)

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


class GpuBackend:
    """A pluggable backend that provides GPU discovery and metric collection."""

    name = "gpu-base"

    def __init__(self, monitor: Optional[MonitorProtocol] = None):
        self._monitor = monitor

    def setup(self) -> None:
        """Initialize backend and attach any discovered handles to the monitor."""
        return None

    def shutdown(self) -> None:
        """Clean up resources if needed."""
        return None

    def _iter_handles(self) -> Iterable[object]:
        return []

    def _collect_system(self, handle: object) -> tuple[float, float, float]:
        raise NotImplementedError

    def _collect_process(self, handle: object) -> tuple[float, float, float]:
        raise NotImplementedError

    def _collect_other(
        self, handle: object, level: str
    ) -> tuple[float, float, float]:
        raise NotImplementedError

    def collect(self, level: str = "process"):
        """Collect metrics for the given level.

        Returns: (gpu_util, gpu_band, gpu_mem)
        """
        gpu_util, gpu_band, gpu_mem = [], [], []

        for handle in self._iter_handles():
            if level == "system":
                util, band, mem = self._collect_system(handle)
            elif level == "process":
                util, band, mem = self._collect_process(handle)
            else:  # user or slurm
                util, band, mem = self._collect_other(handle, level)
            gpu_util.append(util)
            gpu_band.append(band)
            gpu_mem.append(mem)

        return gpu_util, gpu_band, gpu_mem


class NullGpuBackend(GpuBackend):
    """A no-op backend used when no GPU backend is available."""

    name = "gpu-disabled"

    def _iter_handles(self):
        return []


class NvmlGpuBackend(GpuBackend):
    """NVIDIA NVML backend (uses pynvml)."""

    name = "nvidia-nvml"

    def __init__(self, monitor: "PerformanceMonitor"):
        super().__init__(monitor)
        self._pynvml = None

    def _iter_handles(self):
        return self._monitor.nvidia_gpu_handles

    def _get_util_rates(self, handle):
        if self._pynvml is None:
            class DefaultUtilRates:
                gpu = 0.0
                memory = 0.0

            return DefaultUtilRates()
        try:
            return self._pynvml.nvmlDeviceGetUtilizationRates(handle)
        except self._pynvml.NVMLError:
            # If permission denied or other error, use default values
            class DefaultUtilRates:
                gpu = 0.0
                memory = 0.0

            return DefaultUtilRates()

    def setup(self) -> None:
        # Logic is intentionally kept identical to the previous implementation.
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            globals()["pynvml"] = pynvml
            ngpus = self._pynvml.nvmlDeviceGetCount()
            self._monitor.nvidia_gpu_handles = [
                self._pynvml.nvmlDeviceGetHandleByIndex(i)
                for i in range(ngpus)
            ]
            if self._monitor.nvidia_gpu_handles:
                handle = self._monitor.nvidia_gpu_handles[0]
                gpu_mem = round(
                    self._pynvml.nvmlDeviceGetMemoryInfo(handle).total
                    / (1024**3),
                    2,
                )
                if self._monitor.gpu_memory == 0:
                    self._monitor.gpu_memory = gpu_mem
                name = self._pynvml.nvmlDeviceGetName(handle)
                gpu_name = name.decode() if isinstance(name, bytes) else name
                if not self._monitor.gpu_name:
                    self._monitor.gpu_name = gpu_name
                else:
                    self._monitor.gpu_name += f", {gpu_name}"
        except ImportError:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.PYNVML_NOT_AVAILABLE
                ]
            )
            self._monitor.nvidia_gpu_handles = []
        except Exception:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.NVIDIA_DRIVERS_NOT_AVAILABLE
                ]
            )
            self._monitor.nvidia_gpu_handles = []

    def _collect_system(self, handle):
        util_rates = self._get_util_rates(handle)
        memory_info = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
        return util_rates.gpu, 0.0, memory_info.used / (1024**3)

    def _collect_process(self, handle):
        util_rates = self._get_util_rates(handle)
        pids = self._monitor.process_pids
        process_mem = (
            sum(
                p.usedGpuMemory
                for p in self._pynvml.nvmlDeviceGetComputeRunningProcesses(
                    handle
                )
                if p.pid in pids and p.usedGpuMemory
            )
            / (1024**3)
        )
        return util_rates.gpu if process_mem > 0 else 0.0, 0.0, process_mem

    def _collect_other(self, handle, level: str):
        util_rates = self._get_util_rates(handle)
        filtered_gpu_processes, all_processes = (
            self._monitor._get_filtered_processes(level, "nvidia_gpu", handle)
        )
        filtered_mem = (
            sum(
                p.usedGpuMemory
                for p in filtered_gpu_processes
                if p.usedGpuMemory
            )
            / (1024**3)
        )
        filtered_util = (
            (
                util_rates.gpu
                * len(filtered_gpu_processes)
                / max(len(all_processes), 1)
            )
            if filtered_gpu_processes
            else 0.0
        )
        return filtered_util, 0.0, filtered_mem

    def shutdown(self) -> None:
        return None


class AdlxGpuBackend(GpuBackend):
    """AMD ADLX backend (uses ADLXPybind)."""

    name = "amd-adlx"

    def __init__(self, monitor: "PerformanceMonitor"):
        super().__init__(monitor)
        self._adlx_helper = None
        self._adlx_system = None

    def _iter_handles(self):
        return self._monitor.amd_gpu_handles

    def setup(self) -> None:
        # Logic is intentionally kept identical to the previous implementation.
        try:
            from ADLXPybind import ADLXHelper, ADLX_RESULT

            self._adlx_helper = ADLXHelper()
            if self._adlx_helper.Initialize() != ADLX_RESULT.ADLX_OK:
                self._monitor.amd_gpu_handles = []
                return
            self._adlx_system = self._adlx_helper.GetSystemServices()
            gpus_list = self._adlx_system.GetGPUs()
            num_amd_gpus = gpus_list.Size()
            self._monitor.amd_gpu_handles = [
                gpus_list.At(i) for i in range(num_amd_gpus)
            ]
            if self._monitor.amd_gpu_handles:
                gpu = self._monitor.amd_gpu_handles[0]
                # Get memory info
                gpu_mem_info = gpu.TotalVRAM()
                gpu_mem = round(gpu_mem_info / (1024**3), 2)
                if self._monitor.gpu_memory == 0:
                    self._monitor.gpu_memory = gpu_mem
                # Get GPU name
                gpu_name = gpu.Name()
                if not self._monitor.gpu_name:
                    self._monitor.gpu_name = gpu_name
                else:
                    self._monitor.gpu_name += f", {gpu_name}"
        except ImportError:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.ADLX_NOT_AVAILABLE
                ]
            )
            self._monitor.amd_gpu_handles = []
        except Exception:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.AMD_DRIVERS_NOT_AVAILABLE
                ]
            )
            self._monitor.amd_gpu_handles = []

    def _collect_system(self, handle):
        try:
            if self._adlx_system is None:
                return 0.0, 0.0, 0.0
            # Get performance metrics interface
            perf_monitoring = (
                self._adlx_system.GetPerformanceMonitoringServices()
            )

            # Get current metrics
            current_metrics = perf_monitoring.GetCurrentPerformanceMetrics(
                handle
            )

            # Get GPU utilization
            util = current_metrics.GPUUsage()

            # Get memory info
            mem_info = current_metrics.GPUVRAMUsage()

            # AMD ADLX doesn't provide memory bandwidth easily
            return util, 0.0, mem_info / 1024.0
        except Exception:
            # If we can't get metrics, return zeros
            return 0.0, 0.0, 0.0

    def _collect_process(self, handle):
        # AMD ADLX doesn't provide per-process metrics easily
        return 0.0, 0.0, 0.0

    def _collect_other(self, handle, level: str):
        # AMD ADLX doesn't provide per-user metrics easily
        return 0.0, 0.0, 0.0

    def shutdown(self) -> None:
        return None


class ProcessBackend:
    """Backend for process enumeration and filtering."""

    name = "process-base"

    def __init__(self, monitor: "PerformanceMonitor"):
        self._m = monitor

    def setup(self) -> None:
        return None

    def get_process_pids(self) -> set[int]:
        raise NotImplementedError

    def filter_process(self, proc: psutil.Process, mode: str) -> bool:
        raise NotImplementedError

    def get_filtered_processes(
        self,
        level: str = "user",
        mode: str = "cpu",
        handle: Optional[object] = None,
    ):
        raise NotImplementedError

    def safe_proc_call(
        self,
        proc,
        proc_func: Callable[[psutil.Process], Any],
        default=0,
    ):
        raise NotImplementedError


class PsutilProcessBackend(ProcessBackend):
    """Process backend implemented via psutil."""

    name = "process-psutil"

    def get_process_pids(self) -> set[int]:
        """Get current process PID and all its children PIDs."""
        pids = {self._m.pid}
        try:
            pids.update(
                child.pid for child in self._m.process.children(recursive=True)
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return pids

    def filter_process(self, proc: psutil.Process, mode: str) -> bool:
        """Check if process matches the filtering mode."""
        try:
            if mode == "user":
                return proc.uids().real == self._m.uid
            elif mode == "slurm":
                if not is_slurm_available():
                    return False
                return proc.environ().get("SLURM_JOB_ID") == str(
                    self._m.slurm_job
                )
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
        return False

    def get_filtered_processes(
        self,
        level: str = "user",
        mode: str = "cpu",
        handle: Optional[object] = None,
    ):
        """Get filtered processes for CPU or GPU monitoring."""
        if mode == "cpu":
            return [
                proc
                for proc in psutil.process_iter(["pid", "uids"])
                if self.safe_proc_call(
                    proc, lambda p: self.filter_process(p, level), False
                )
            ]
        elif mode == "nvidia_gpu":
            all_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            filtered = [
                p
                for p in all_procs
                if self.safe_proc_call(
                    p.pid,
                    lambda proc: self.filter_process(proc, level),
                    False,
                )
            ]
            return filtered, all_procs
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def safe_proc_call(
        self,
        proc,
        proc_func: Callable[[psutil.Process], Any],
        default=0,
    ):
        """Safely call a process method and return default on error."""
        try:
            if not isinstance(proc, psutil.Process):
                # proc might be a pid. Moved Process creation here to catch
                # exceptions at the same place
                proc = psutil.Process(proc)
            result = proc_func(proc)
            return result if result is not None else default
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            return default
        except TypeError:
            # in test case, where psutil is a mock
            if isinstance(psutil.Process, unittest.mock.MagicMock):
                return default


class CpuBackend:
    """Backend for CPU metrics."""

    name = "cpu-base"

    def __init__(self, monitor: "PerformanceMonitor"):
        self._m = monitor

    def setup(self) -> None:
        return None

    def collect(self, level: str = "process") -> list[float]:
        raise NotImplementedError


class PsutilCpuBackend(CpuBackend):
    """CPU backend implemented via psutil."""

    name = "cpu-psutil"

    def collect(self, level: str = "process") -> list[float]:
        self._m._validate_level(level)
        if level == "system":
            # just return the whole system here
            cpu_util_per_core = psutil.cpu_percent(percpu=True)
            return cpu_util_per_core
        elif level == "process":
            # get process pids
            pids = self._m.process_pids
            cpu_total = sum(
                self._m._process_backend.safe_proc_call(
                    pid, lambda p: p.cpu_percent(interval=0.1)
                )
                for pid in pids
            )
            return [cpu_total / self._m.num_cpus] * self._m.num_cpus
        else:  # user or slurm
            cpu_total = sum(
                self._m._process_backend.safe_proc_call(
                    proc, lambda p: p.cpu_percent()
                )
                for proc in self._m._process_backend.get_filtered_processes(
                    level, "cpu"
                )
            )
            return [cpu_total / self._m.num_cpus] * self._m.num_cpus


class MemoryBackend:
    """Backend for memory metrics."""

    name = "memory-base"

    def __init__(self, monitor: "PerformanceMonitor"):
        self._m = monitor

    def setup(self) -> None:
        return None

    def collect(self, level: str = "process") -> float:
        raise NotImplementedError


class PsutilMemoryBackend(MemoryBackend):
    """Memory backend implemented via psutil."""

    name = "memory-psutil"

    def collect(self, level: str = "process") -> float:
        self._m._validate_level(level)
        if level == "system":
            return (
                psutil.virtual_memory().total
                - psutil.virtual_memory().available
            ) / (1024**3)
        elif level == "process":
            pids = self._m.process_pids
            memory_total = sum(
                self._m._process_backend.safe_proc_call(
                    pid, lambda p: p.memory_full_info().uss
                )
                for pid in pids
            )
            return memory_total / (1024**3)
        else:  # user or slurm
            memory_total = sum(
                self._m._process_backend.safe_proc_call(
                    proc, lambda p: p.memory_full_info().uss, 0
                )
                for proc in self._m._process_backend.get_filtered_processes(
                    level, "cpu"
                )
            )
            return memory_total / (1024**3)


class IoBackend:
    """Backend for I/O metrics."""

    name = "io-base"

    def __init__(self, monitor: "PerformanceMonitor"):
        self._m = monitor

    def setup(self) -> None:
        return None

    def collect(self, level: str = "process") -> list[int]:
        raise NotImplementedError


class PsutilIoBackend(IoBackend):
    """I/O backend implemented via psutil."""

    name = "io-psutil"

    def collect(self, level: str = "process") -> list[int]:
        self._m._validate_level(level)
        totals = [0, 0, 0, 0]
        if level == "process":
            pids = self._m.process_pids
            for pid in pids:
                io_data = self._m._process_backend.safe_proc_call(
                    pid, lambda p: p.io_counters()
                )
                if io_data:
                    totals[0] += io_data.read_count
                    totals[1] += io_data.write_count
                    totals[2] += io_data.read_bytes
                    totals[3] += io_data.write_bytes
        elif level == "system":
            for proc in psutil.process_iter(["pid"]):
                io_data = self._m._process_backend.safe_proc_call(
                    proc, lambda p: p.io_counters()
                )
                if io_data:
                    totals[0] += io_data.read_count
                    totals[1] += io_data.write_count
                    totals[2] += io_data.read_bytes
                    totals[3] += io_data.write_bytes
        else:  # user or slurm
            for proc in self._m._process_backend.get_filtered_processes(
                level, "cpu"
            ):
                io_data = self._m._process_backend.safe_proc_call(
                    proc, lambda p: p.io_counters()
                )
                if io_data:
                    totals[0] += io_data.read_count
                    totals[1] += io_data.write_count
                    totals[2] += io_data.read_bytes
                    totals[3] += io_data.write_bytes
        return totals


class GpuBackendDiscovery:
    """Selects GPU backends based on what's available at runtime."""

    def __init__(self, monitor: "PerformanceMonitor"):
        self._monitor = monitor

    def discover(self):
        return [
            NvmlGpuBackend(self._monitor),
            AdlxGpuBackend(self._monitor),
        ]


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
