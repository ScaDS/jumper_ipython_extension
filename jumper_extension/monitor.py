import os
import threading
import time

import psutil

from .data import PerformanceData

# GPU monitoring setup
PYNVML_AVAILABLE = False
try:
    import pynvml

    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except ImportError:
    print("[JUmPER]: Warning: pynvml not available. GPU monitoring disabled.")
except Exception:
    print("NVIDIA drivers not available. GPU monitoring disabled.")


class PerformanceMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.running = False
        self.start_time = None
        self.monitor_thread = None
        self.process = psutil.Process()
        self.cpu_handles = self.process.cpu_affinity()
        self.num_cpus = len(self.cpu_handles)
        self.pid = os.getpid()
        self.uid = os.getuid()
        self.slurm_job = os.environ.get("SLURM_JOB_ID", 0)

        self.memory = self._detect_memory_limit()

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

        self.data = PerformanceData(self.num_cpus, self.num_gpus)

    def _detect_memory_limit(self):
        slurm_path = f"/sys/fs/cgroup/memory/slurm/uid_{self.uid}/job_{os.environ.get('SLURM_JOB_ID', 0)}/memory.limit_in_bytes"
        if os.path.exists(slurm_path):
            with open(slurm_path) as f:
                return round(int(f.read().strip()) / (1024**3), 2)
        return round(psutil.virtual_memory().total / (1024**3), 2)

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
                self.gpu_name = name.decode() if isinstance(name, bytes) else name
        except Exception:
            self.gpu_handles = []

    def _validate_level(self, level):
        if level not in ["user", "slurm", "process", "system"]:
            raise ValueError(f"Unknown level: {level}")

    def _filter_process(self, proc, mode):
        """Check if process matches the filtering mode"""
        try:
            if mode == "user":
                return proc.uids().real == self.uid
            elif mode == "slurm":
                return proc.environ().get("SLURM_JOB_ID") == str(self.slurm_job)
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
                    psutil.Process(p.pid),
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
            result = proc_func(proc)
            return result if result is not None else default
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return default

    def _collect_cpu(self, level="process"):
        self._validate_level(level)
        if level == "system":
            cpu_util_per_core = psutil.cpu_percent(percpu=True)
            return [cpu_util_per_core[i] for i in self.cpu_handles]
        elif level == "process":
            process_cpu = self.process.cpu_percent()
            return [process_cpu / self.num_cpus] * self.num_cpus
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
                psutil.virtual_memory().total - psutil.virtual_memory().available
            ) / (1024**3)
        elif level == "process":
            return self.process.memory_full_info().uss / (1024**3)
        else:  # user or slurm
            memory_total = sum(
                self._safe_proc_call(proc, lambda p: p.memory_full_info().uss)
                for proc in self._get_filtered_processes(level, "cpu")
            )
            return memory_total / (1024**3)

    def _collect_io(self, level="process"):
        self._validate_level(level)
        if level == "process":
            io_data = self.process.io_counters()
            return [
                io_data.read_count,
                io_data.write_count,
                io_data.read_bytes / (1024**2),
                io_data.write_bytes / (1024**2),
            ]
        elif level == "system":
            try:
                disk_io = psutil.disk_io_counters()
                return (
                    [
                        disk_io.read_count,
                        disk_io.write_count,
                        disk_io.read_bytes / (1024**2),
                        disk_io.write_bytes / (1024**2),
                    ]
                    if disk_io
                    else [0, 0, 0.0, 0.0]
                )
            except Exception:
                return [0, 0, 0.0, 0.0]
        else:  # user or slurm
            totals = [0, 0, 0, 0]  # read_count, write_count, read_bytes, write_bytes
            for proc in self._get_filtered_processes(level, "cpu"):
                io_data = self._safe_proc_call(proc, lambda p: p.io_counters())
                if io_data:
                    totals[0] += io_data.read_count
                    totals[1] += io_data.write_count
                    totals[2] += io_data.read_bytes
                    totals[3] += io_data.write_bytes
            return [totals[0], totals[1], totals[2] / (1024**2), totals[3] / (1024**2)]

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
                process_mem = sum(
                    p.usedGpuMemory
                    for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    if p.pid == self.pid
                ) / (1024**3)
                gpu_util.append(util_rates.gpu if process_mem > 0 else 0.0)
                gpu_band.append(0.0)
                gpu_mem.append(process_mem)
            else:  # user or slurm
                filtered_gpu_processes, all_processes = self._get_filtered_processes(
                    level, "gpu", handle
                )
                filtered_mem = sum(p.usedGpuMemory for p in filtered_gpu_processes) / (
                    1024**3
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
                gpu_util.append(filtered_util)
                gpu_band.append(0.0)
                gpu_mem.append(filtered_mem)

        return gpu_util, gpu_band, gpu_mem

    def _collect_metrics(self):
        time_mark = time.time()
        return tuple(
            (
                time_mark,
                self._collect_cpu(level),
                self._collect_memory(level),
                *self._collect_gpu(level),
                self._collect_io(level),
            )
            for level in ["user", "process", "system", "slurm"]
        )

    def _collect_data(self):
        levels = ["user", "process", "system", "slurm"]
        while self.running:
            metrics = self._collect_metrics()
            for level, data_tuple in zip(levels, metrics):
                self.data.add_sample(level, *data_tuple)
            time.sleep(self.interval)

    def start(self):
        if self.running:
            print("[JUmPER]: Performance monitor already running")
            return
        self.start_time = time.time()
        self.running = True
        self.monitor_thread = threading.Thread(target=self._collect_data, daemon=True)
        self.monitor_thread.start()
        print(
            f"[JUmPER]: Performance monitoring started "
            f"(PID: {self.pid}, Interval: {self.interval}s)"
        )

    def stop(self):
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        print(
            f"[JUmPER]: Performance monitoring stopped "
            f"(ran for {time.time() - self.start_time:.2f} seconds)"
        )
