import logging

from jumper_extension.core.messages import (
    ExtensionErrorCode,
    EXTENSION_ERROR_MESSAGES,
)
from jumper_extension.monitor.metrics.gpu.common import GpuBackend

logger = logging.getLogger("extension")


class NvmlGpuBackend(GpuBackend):
    """NVIDIA NVML backend (uses pynvml)."""

    name = "nvidia-nvml"

    def __init__(self, monitor):
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
