from typing import Iterable

import psutil

from jumper_extension.utilities import is_slurm_available
from jumper_extension.monitor.metrics.context import CollectionContext


class GpuBackend:
    """A pluggable backend that provides GPU discovery and metric collection."""

    name = "gpu-base"

    def __init__(self, uid: int, slurm_job: str):
        self._uid = uid
        self._slurm_job = slurm_job

    def setup(self) -> None:
        """Initialize backend and discover GPU handles."""
        return None

    def shutdown(self) -> None:
        """Clean up resources if needed."""
        return None

    def _iter_handles(self) -> Iterable[object]:
        return []

    def _filter_process(self, pid: int, mode: str) -> bool:
        """Filter a GPU process by user/slurm membership."""
        try:
            proc = psutil.Process(pid)
            if mode == "user":
                return proc.uids().real == self._uid
            elif mode == "slurm":
                if not is_slurm_available():
                    return False
                return proc.environ().get("SLURM_JOB_ID") == str(self._slurm_job)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
        return False

    def _collect_system(self, handle: object) -> tuple[float, float, float]:
        raise NotImplementedError

    def _collect_process(
        self, handle: object, context: CollectionContext
    ) -> tuple[float, float, float]:
        raise NotImplementedError

    def _collect_other(
        self, handle: object, level: str, context: CollectionContext
    ) -> tuple[float, float, float]:
        raise NotImplementedError

    def snapshot(self, context: CollectionContext) -> None:
        return None

    def collect(self, level: str, context: CollectionContext):
        """Collect metrics for the given level.

        Returns: (gpu_util, gpu_band, gpu_mem)
        """
        gpu_util, gpu_band, gpu_mem = [], [], []

        for handle in self._iter_handles():
            if level == "system":
                util, band, mem = self._collect_system(handle)
            elif level == "process":
                util, band, mem = self._collect_process(handle, context)
            else:  # user or slurm
                util, band, mem = self._collect_other(handle, level, context)
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

    def __init__(self, available: dict):
        self._available = available

    def discover(self):
        from jumper_extension.monitor.metrics.gpu.nvml import NvmlGpuBackend
        from jumper_extension.monitor.metrics.gpu.adlx import AdlxGpuBackend
        from jumper_extension.config.utils import instantiate_backend

        return [
            instantiate_backend(NvmlGpuBackend, self._available),
            instantiate_backend(AdlxGpuBackend, self._available),
        ]
