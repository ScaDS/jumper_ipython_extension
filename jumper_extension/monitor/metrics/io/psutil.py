import psutil

from jumper_extension.monitor.metrics.io.common import IoBackend


class PsutilIoBackend(IoBackend):
    """I/O backend implemented via psutil."""

    name = "io-psutil"

    def collect(self, level: str = "process") -> list[int]:
        self._m._validate_level(level)
        totals = [0, 0, 0, 0]
        if level == "process":
            pids = self._m.process_pids
            for pid in pids:
                io_data = self._m._process_backend.safe_proc_call(
                    pid, lambda p: p.io_counters()
                )
                if io_data:
                    totals[0] += io_data.read_count
                    totals[1] += io_data.write_count
                    totals[2] += io_data.read_bytes
                    totals[3] += io_data.write_bytes
        elif level == "system":
            for proc in psutil.process_iter(["pid"]):
                io_data = self._m._process_backend.safe_proc_call(
                    proc, lambda p: p.io_counters()
                )
                if io_data:
                    totals[0] += io_data.read_count
                    totals[1] += io_data.write_count
                    totals[2] += io_data.read_bytes
                    totals[3] += io_data.write_bytes
        else:  # user or slurm
            for proc in self._m._process_backend.get_filtered_processes(
                level, "cpu"
            ):
                io_data = self._m._process_backend.safe_proc_call(
                    proc, lambda p: p.io_counters()
                )
                if io_data:
                    totals[0] += io_data.read_count
                    totals[1] += io_data.write_count
                    totals[2] += io_data.read_bytes
                    totals[3] += io_data.write_bytes
        return totals
