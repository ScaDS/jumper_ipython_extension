import psutil

from jumper_extension.monitor.metrics.cpu.common import CpuBackend


class PsutilCpuBackend(CpuBackend):
    """CPU backend implemented via psutil."""

    name = "cpu-psutil"

    def collect(self, level: str = "process") -> list[float]:
        self._m._validate_level(level)
        if level == "system":
            # just return the whole system here
            cpu_util_per_core = psutil.cpu_percent(percpu=True)
            return cpu_util_per_core
        elif level == "process":
            # get process pids
            pids = self._m.process_pids
            cpu_total = sum(
                self._m._process_backend.safe_proc_call(
                    pid, lambda p: p.cpu_percent(interval=0.1)
                )
                for pid in pids
            )
            return [cpu_total / self._m.num_cpus] * self._m.num_cpus
        else:  # user or slurm
            cpu_total = sum(
                self._m._process_backend.safe_proc_call(
                    proc, lambda p: p.cpu_percent()
                )
                for proc in self._m._process_backend.get_filtered_processes(
                    level, "cpu"
                )
            )
            return [cpu_total / self._m.num_cpus] * self._m.num_cpus
