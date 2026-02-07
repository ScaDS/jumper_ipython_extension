from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jumper_extension.monitor.common import PerformanceMonitor


class MemoryBackend:
    """Backend for memory metrics."""

    name = "memory-base"

    def __init__(self, monitor: "PerformanceMonitor"):
        self._m = monitor

    def setup(self) -> None:
        return None

    def collect(self, level: str = "process") -> float:
        raise NotImplementedError
