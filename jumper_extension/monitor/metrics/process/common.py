from typing import Any, Callable, Optional

import psutil

from jumper_extension.monitor.metrics.context import CollectionContext


class ProcessCollectorBackend:
    """Backend for process enumeration and snapshotting."""

    name = "process-base"

    def __init__(self, pid: int, process: psutil.Process, uid: int, slurm_job: str):
        self._pid = pid
        self._process = process
        self._uid = uid
        self._slurm_job = slurm_job

    def setup(self) -> None:
        return None

    def get_process_pids(self) -> set[int]:
        raise NotImplementedError

    def collect(self, level: str, context: CollectionContext) -> None:
        raise NotImplementedError

    def snapshot(self, context: CollectionContext) -> None:
        raise NotImplementedError

    def filter_process(self, proc: psutil.Process, mode: str) -> bool:
        raise NotImplementedError

    def get_filtered_processes(
        self,
        level: str = "user",
        mode: str = "cpu",
        handle: Optional[object] = None,
    ) -> list[psutil.Process]:
        raise NotImplementedError

    def safe_proc_call(
        self,
        proc,
        proc_func: Callable[[psutil.Process], Any],
        default=0,
    ) -> Any:
        raise NotImplementedError
