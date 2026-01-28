import logging

from jumper_extension.core.messages import (
    ExtensionErrorCode,
    EXTENSION_ERROR_MESSAGES,
)
from jumper_extension.monitor.metrics.gpu.common import GpuBackend

logger = logging.getLogger("extension")


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
