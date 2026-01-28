from typing import Iterable, Optional, TYPE_CHECKING

# To avoid circular imports
if TYPE_CHECKING:
    from jumper_extension.monitor.common import MonitorProtocol


class GpuBackend:
    """A pluggable backend that provides GPU discovery and metric collection."""

    name = "gpu-base"

    def __init__(self, monitor: Optional["MonitorProtocol"] = None):
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


class GpuBackendDiscovery:
    """Selects GPU backends based on what's available at runtime."""

    def __init__(self, monitor):
        self._monitor = monitor

    def discover(self):
        from jumper_extension.monitor.metrics.gpu.nvml import NvmlGpuBackend
        from jumper_extension.monitor.metrics.gpu.adlx import AdlxGpuBackend

        return [
            NvmlGpuBackend(self._monitor),
            AdlxGpuBackend(self._monitor),
        ]
