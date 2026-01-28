import unittest.mock
from typing import Any, Callable, Optional

import psutil

from jumper_extension.utilities import is_slurm_available
from jumper_extension.monitor.metrics.process.common import ProcessBackend


class PsutilProcessBackend(ProcessBackend):
    """Process backend implemented via psutil."""

    name = "process-psutil"

    def get_process_pids(self) -> set[int]:
        """Get current process PID and all its children PIDs."""
        pids = {self._m.pid}
        try:
            pids.update(
                child.pid for child in self._m.process.children(recursive=True)
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return pids

    def filter_process(self, proc: psutil.Process, mode: str) -> bool:
        """Check if process matches the filtering mode."""
        try:
            if mode == "user":
                return proc.uids().real == self._m.uid
            elif mode == "slurm":
                if not is_slurm_available():
                    return False
                return proc.environ().get("SLURM_JOB_ID") == str(
                    self._m.slurm_job
                )
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
        return False

    def get_filtered_processes(
        self,
        level: str = "user",
        mode: str = "cpu",
        handle: Optional[object] = None,
    ):
        """Get filtered processes for CPU or GPU monitoring."""
        if mode == "cpu":
            return [
                proc
                for proc in psutil.process_iter(["pid", "uids"])
                if self.safe_proc_call(
                    proc, lambda p: self.filter_process(p, level), False
                )
            ]
        elif mode == "nvidia_gpu":
            try:
                import pynvml
            except ImportError:
                return [], []
            all_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            filtered = [
                p
                for p in all_procs
                if self.safe_proc_call(
                    p.pid,
                    lambda proc: self.filter_process(proc, level),
                    False,
                )
            ]
            return filtered, all_procs
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def safe_proc_call(
        self,
        proc,
        proc_func: Callable[[psutil.Process], Any],
        default=0,
    ):
        """Safely call a process method and return default on error."""
        try:
            if not isinstance(proc, psutil.Process):
                # proc might be a pid. Moved Process creation here to catch
                # exceptions at the same place
                proc = psutil.Process(proc)
            result = proc_func(proc)
            return result if result is not None else default
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            return default
        except TypeError:
            # in test case, where psutil is a mock
            if isinstance(psutil.Process, unittest.mock.MagicMock):
                return default
