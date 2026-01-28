import psutil

from jumper_extension.monitor.metrics.memory.common import MemoryBackend


class PsutilMemoryBackend(MemoryBackend):
    """Memory backend implemented via psutil."""

    name = "memory-psutil"

    def collect(self, level: str = "process") -> float:
        self._m._validate_level(level)
        if level == "system":
            return (
                psutil.virtual_memory().total
                - psutil.virtual_memory().available
            ) / (1024**3)
        elif level == "process":
            pids = self._m.process_pids
            memory_total = sum(
                self._m._process_backend.safe_proc_call(
                    pid, lambda p: p.memory_full_info().uss
                )
                for pid in pids
            )
            return memory_total / (1024**3)
        else:  # user or slurm
            memory_total = sum(
                self._m._process_backend.safe_proc_call(
                    proc, lambda p: p.memory_full_info().uss, 0
                )
                for proc in self._m._process_backend.get_filtered_processes(
                    level, "cpu"
                )
            )
            return memory_total / (1024**3)
