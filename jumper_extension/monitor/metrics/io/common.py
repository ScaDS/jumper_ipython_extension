from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jumper_extension.monitor.common import PerformanceMonitor


class IoBackend:
    """Backend for I/O metrics."""

    name = "io-base"

    def __init__(self, monitor: "PerformanceMonitor"):
        self._m = monitor

    def setup(self) -> None:
        return None

    def collect(self, level: str = "process") -> list[int]:
        raise NotImplementedError
