from typing import Any, Callable, Optional, TYPE_CHECKING

import psutil

if TYPE_CHECKING:
    from jumper_extension.monitor.common import PerformanceMonitor


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
