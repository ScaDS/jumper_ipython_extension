import logging
import os
import threading
import time
import unittest.mock

import psutil

from .data import PerformanceData
from .extension_messages import (
    ExtensionErrorCode,
    ExtensionInfoCode,
    EXTENSION_ERROR_MESSAGES,
    EXTENSION_INFO_MESSAGES,
)
from .utilities import (
    get_available_levels,
    is_slurm_available,
    detect_memory_limit,
)

logger = logging.getLogger("extension")

# GPU monitoring setup
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


class PerformanceMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.running = False
        self.start_time = None
        self.monitor_thread = None
        self.process = psutil.Process()
        self.cpu_handles = self.process.cpu_affinity()
        self.num_cpus = len(self.cpu_handles)
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

        self.gpu_handles = []
        self.gpu_memory = 0
        self.gpu_name = ""
        if PYNVML_AVAILABLE:
            self._setup_gpu()
        self.num_gpus = len(self.gpu_handles)

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

    def _setup_gpu(self):
        try:
            ngpus = pynvml.nvmlDeviceGetCount()
            self.gpu_handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(ngpus)
            ]
            if self.gpu_handles:
                handle = self.gpu_handles[0]
                self.gpu_memory = round(
                    pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**3), 2
                )
                name = pynvml.nvmlDeviceGetName(handle)
                self.gpu_name = (
                    name.decode() if isinstance(name, bytes) else name
                )
        except Exception:
            self.gpu_handles = []

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
        elif mode == "gpu":
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
        except (psutil.NoSuchProcess, psutil.AccessDenied):
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
        if not PYNVML_AVAILABLE or not self.gpu_handles:
            return [], [], []

        self._validate_level(level)
        gpu_util, gpu_band, gpu_mem = [], [], []

        for handle in self.gpu_handles:
            util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)

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
                    self._get_filtered_processes(level, "gpu", handle)
                )
                filtered_mem = sum(
                    p.usedGpuMemory for p in filtered_gpu_processes if p.usedGpuMemory
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
            if time_measurement > self.interval:
                print(
                    EXTENSION_INFO_MESSAGES[
                        ExtensionInfoCode.IMPRECISE_INTERVAL
                    ].format(interval=self.interval),
                    end="\r",
                )
            else:
                time.sleep(self.interval - time_measurement)

    def start(self):
        if self.running:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.MONITOR_ALREADY_RUNNING
                ]
            )
            return
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
        logger.info(
            EXTENSION_INFO_MESSAGES[ExtensionInfoCode.MONITOR_STOPPED].format(
                seconds=time.perf_counter() - self.start_time
            )
        )
