import logging
import os
import threading
import time
import unittest.mock
from typing import Protocol, Optional, runtime_checkable

import psutil

from jumper_extension.core.data import PerformanceData
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

# NVIDIA GPU monitoring setup
PYNVML_AVAILABLE = False
try:
    import pynvml

    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except ImportError:
    logger.warning(
        EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.PYNVML_NOT_AVAILABLE]
    )
except Exception:
    logger.warning(
        EXTENSION_ERROR_MESSAGES[
            ExtensionErrorCode.NVIDIA_DRIVERS_NOT_AVAILABLE
        ]
    )

# AMD GPU monitoring setup
ADLX_AVAILABLE = False
try:
    from ADLXPybind import (
        ADLX,
        ADLXHelper,
        ADLX_RESULT,
    )

    adlx_helper = ADLXHelper()
    if adlx_helper.Initialize() == ADLX_RESULT.ADLX_OK:
        ADLX_AVAILABLE = True
        adlx_system = adlx_helper.GetSystemServices()
except ImportError:
    logger.warning(
        EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.ADLX_NOT_AVAILABLE]
    )
except Exception:
    logger.warning(
        EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.AMD_DRIVERS_NOT_AVAILABLE]
    )

@runtime_checkable
class PerformanceMonitorProtocol(Protocol):
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

        self.nvidia_gpu_handles = []
        self.amd_gpu_handles = []
        self.gpu_memory = 0
        self.gpu_name = ""
        if PYNVML_AVAILABLE:
            self._setup_nvidia_gpu()
        if ADLX_AVAILABLE:
            self._setup_amd_gpu()
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

    def _setup_nvidia_gpu(self):
        try:
            ngpus = pynvml.nvmlDeviceGetCount()
            self.nvidia_gpu_handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(ngpus)
            ]
            if self.nvidia_gpu_handles:
                handle = self.nvidia_gpu_handles[0]
                gpu_mem = round(
                    pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**3), 2
                )
                if self.gpu_memory == 0:
                    self.gpu_memory = gpu_mem
                name = pynvml.nvmlDeviceGetName(handle)
                gpu_name = name.decode() if isinstance(name, bytes) else name
                if not self.gpu_name:
                    self.gpu_name = gpu_name
                else:
                    self.gpu_name += f", {gpu_name}"
        except Exception:
            self.nvidia_gpu_handles = []

    def _setup_amd_gpu(self):
        try:
            gpus_list = adlx_system.GetGPUs()
            num_amd_gpus = gpus_list.Size()
            self.amd_gpu_handles = [
                gpus_list.At(i) for i in range(num_amd_gpus)
            ]
            if self.amd_gpu_handles:
                gpu = self.amd_gpu_handles[0]
                # Get memory info
                gpu_mem_info = gpu.TotalVRAM()
                gpu_mem = round(gpu_mem_info / (1024**3), 2)
                if self.gpu_memory == 0:
                    self.gpu_memory = gpu_mem
                # Get GPU name
                gpu_name = gpu.Name()
                if not self.gpu_name:
                    self.gpu_name = gpu_name
                else:
                    self.gpu_name += f", {gpu_name}"
        except Exception:
            self.amd_gpu_handles = []

    def _get_process_pids(self):
        """Get current process PID and all its children PIDs"""
        pids = {self.pid}
        try:
            pids.update(
                child.pid for child in self.process.children(recursive=True)
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return pids

    def _validate_level(self, level):
        if level not in self.levels:
            raise ValueError(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.INVALID_LEVEL
                ].format(level=level, levels=self.levels)
            )

    def _filter_process(self, proc, mode):
        """Check if process matches the filtering mode"""
        try:
            if mode == "user":
                return proc.uids().real == self.uid
            elif mode == "slurm":
                if not is_slurm_available():
                    return False
                return proc.environ().get("SLURM_JOB_ID") == str(
                    self.slurm_job
                )
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
        return False

    def _get_filtered_processes(self, level="user", mode="cpu", handle=None):
        """Get filtered processes for CPU or GPU monitoring"""
        if mode == "cpu":
            return [
                proc
                for proc in psutil.process_iter(["pid", "uids"])
                if self._safe_proc_call(
                    proc, lambda p: self._filter_process(p, level), False
                )
            ]
        elif mode == "nvidia_gpu":
            all_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            filtered = [
                p
                for p in all_procs
                if self._safe_proc_call(
                    p.pid,
                    lambda proc: self._filter_process(proc, level),
                    False,
                )
            ]
            return filtered, all_procs
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _safe_proc_call(self, proc, proc_func, default=0):
        """Safely call a process method and return default on error"""
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

    def _collect_cpu(self, level="process"):
        self._validate_level(level)
        if level == "system":
            # just return the whole system here
            cpu_util_per_core = psutil.cpu_percent(percpu=True)
            return cpu_util_per_core
            # return [cpu_util_per_core[i] for i in self.cpu_handles]
        elif level == "process":
            # get process pids
            pids = self.process_pids
            cpu_total = sum(
                self._safe_proc_call(
                    pid, lambda p: p.cpu_percent(interval=0.1)
                )
                for pid in pids
            )
            return [cpu_total / self.num_cpus] * self.num_cpus
        else:  # user or slurm
            cpu_total = sum(
                self._safe_proc_call(proc, lambda p: p.cpu_percent())
                for proc in self._get_filtered_processes(level, "cpu")
            )
            return [cpu_total / self.num_cpus] * self.num_cpus

    def _collect_memory(self, level="process"):
        self._validate_level(level)
        if level == "system":
            return (
                psutil.virtual_memory().total
                - psutil.virtual_memory().available
            ) / (1024**3)
        elif level == "process":
            pids = self.process_pids
            memory_total = sum(
                self._safe_proc_call(pid, lambda p: p.memory_full_info().uss)
                for pid in pids
            )
            return memory_total / (1024**3)
        else:  # user or slurm
            memory_total = sum(
                self._safe_proc_call(
                    proc, lambda p: p.memory_full_info().uss, 0
                )
                for proc in self._get_filtered_processes(level, "cpu")
            )
            return memory_total / (1024**3)

    def _collect_io(self, level="process"):
        self._validate_level(level)
        if level == "process":
            pids = self.process_pids
            totals = [0, 0, 0, 0]
            for pid in pids:
                io_data = self._safe_proc_call(pid, lambda p: p.io_counters())
                if io_data:
                    totals[0] += io_data.read_count
                    totals[1] += io_data.write_count
                    totals[2] += io_data.read_bytes
                    totals[3] += io_data.write_bytes
        elif level == "system":
            totals = [0, 0, 0, 0]
            for proc in psutil.process_iter(["pid"]):
                io_data = self._safe_proc_call(proc, lambda p: p.io_counters())
                if io_data:
                    totals[0] += io_data.read_count
                    totals[1] += io_data.write_count
                    totals[2] += io_data.read_bytes
                    totals[3] += io_data.write_bytes
        else:  # user or slurm
            totals = [0, 0, 0, 0]
            for proc in self._get_filtered_processes(level, "cpu"):
                io_data = self._safe_proc_call(proc, lambda p: p.io_counters())
                if io_data:
                    totals[0] += io_data.read_count
                    totals[1] += io_data.write_count
                    totals[2] += io_data.read_bytes
                    totals[3] += io_data.write_bytes
        return totals

    def _collect_gpu(self, level="process"):
        if not (PYNVML_AVAILABLE or ADLX_AVAILABLE) or self.num_gpus == 0:
            return [], [], []

        self._validate_level(level)
        gpu_util, gpu_band, gpu_mem = [], [], []

        # Collect NVIDIA GPU metrics
        if PYNVML_AVAILABLE and self.nvidia_gpu_handles:
            nvidia_util, nvidia_band, nvidia_mem = self._collect_nvidia_gpu(
                level
            )
            gpu_util.extend(nvidia_util)
            gpu_band.extend(nvidia_band)
            gpu_mem.extend(nvidia_mem)

        # Collect AMD GPU metrics
        if ADLX_AVAILABLE and self.amd_gpu_handles:
            amd_util, amd_band, amd_mem = self._collect_amd_gpu(level)
            gpu_util.extend(amd_util)
            gpu_band.extend(amd_band)
            gpu_mem.extend(amd_mem)

        return gpu_util, gpu_band, gpu_mem

    def _collect_nvidia_gpu(self, level="process"):
        """Collect NVIDIA GPU metrics"""
        gpu_util, gpu_band, gpu_mem = [], [], []

        for handle in self.nvidia_gpu_handles:
            try:
                util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
            except pynvml.NVMLError:
                # If permission denied or other error, use default values
                class DefaultUtilRates:
                    gpu = 0.0
                    memory = 0.0
                util_rates = DefaultUtilRates()

            if level == "system":
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_util.append(util_rates.gpu)
                gpu_band.append(util_rates.memory)
                gpu_mem.append(memory_info.used / (1024**3))
            elif level == "process":
                pids = self.process_pids
                process_mem = sum(
                    p.usedGpuMemory
                    for p in pynvml.nvmlDeviceGetComputeRunningProcesses(
                        handle
                    )
                    if p.pid in pids and p.usedGpuMemory
                ) / (1024**3)
                gpu_util.append(util_rates.gpu if process_mem > 0 else 0.0)
                gpu_band.append(0.0)
                gpu_mem.append(process_mem)
            else:  # user or slurm
                filtered_gpu_processes, all_processes = (
                    self._get_filtered_processes(level, "nvidia_gpu", handle)
                )
                filtered_mem = sum(
                    p.usedGpuMemory
                    for p in filtered_gpu_processes
                    if p.usedGpuMemory
                ) / (1024**3)
                filtered_util = (
                    (
                        util_rates.gpu
                        * len(filtered_gpu_processes)
                        / max(len(all_processes), 1)
                    )
                    if filtered_gpu_processes
                    else 0.0
                )
                gpu_util.append(filtered_util)
                gpu_band.append(0.0)
                gpu_mem.append(filtered_mem)

        return gpu_util, gpu_band, gpu_mem

    def _collect_amd_gpu(self, level="process"):
        """Collect AMD GPU metrics"""
        gpu_util, gpu_band, gpu_mem = [], [], []

        for gpu_handle in self.amd_gpu_handles:
            try:
                # Get performance metrics interface
                perf_monitoring = (
                    adlx_system.GetPerformanceMonitoringServices()
                )

                # Get current metrics
                current_metrics = perf_monitoring.GetCurrentPerformanceMetrics(
                    gpu_handle
                )

                # Get GPU utilization
                util = current_metrics.GPUUsage()

                # Get memory info
                mem_info = current_metrics.GPUVRAMUsage()

                if level == "system":
                    gpu_util.append(util)
                    gpu_band.append(
                        0.0
                    )  # AMD ADLX doesn't provide memory bandwidth easily
                    # Convert VRAM usage from MB to GB
                    gpu_mem.append(mem_info / 1024.0)
                elif level == "process":
                    # AMD ADLX doesn't provide per-process metrics easily
                    # For now, we'll report 0 for process-level
                    gpu_util.append(0.0)
                    gpu_band.append(0.0)
                    gpu_mem.append(0.0)
                else:  # user or slurm
                    # AMD ADLX doesn't provide per-user metrics easily
                    # For now, we'll report 0 for user/slurm level
                    gpu_util.append(0.0)
                    gpu_band.append(0.0)
                    gpu_mem.append(0.0)
            except Exception:
                # If we can't get metrics, append zeros
                gpu_util.append(0.0)
                gpu_band.append(0.0)
                gpu_mem.append(0.0)

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
