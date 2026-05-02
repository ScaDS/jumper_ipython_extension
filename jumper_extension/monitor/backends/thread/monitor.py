import logging
import os
import threading
import time

import psutil

from jumper_extension.adapters.data import NodeInfo, NodeDataStore
from jumper_extension.config.utils import instantiate, load_collectors_config
from jumper_extension.core.messages import (
    ExtensionErrorCode,
    ExtensionInfoCode,
    EXTENSION_ERROR_MESSAGES,
    EXTENSION_INFO_MESSAGES,
)
from jumper_extension.monitor.metrics.context import CollectionContext
from jumper_extension.monitor.metrics.process.psutil import PsutilProcessBackend
from jumper_extension.monitor.metrics.storage import make_handler
from jumper_extension.utilities import detect_memory_limit, get_available_levels

logger = logging.getLogger("extension")


class PerformanceMonitor:
    def __init__(self):
        self.interval = 1.0
        self.running = False
        self.start_time = None
        self.stop_time = None
        self.wallclock_start_time = None
        self.wallclock_stop_time = None
        self.monitor_thread = None
        self.process = psutil.Process()
        self.n_measurements = 0
        self.n_missed_measurements = 0
        """
        on MacOS cpu_affinity is not implemented in psutil
        (raises AttributeError)
        set the num_cpus to the number of cpus in the system
        same for cpu_affinity
        """
        try:
            cpu_handles = self.process.cpu_affinity()
            num_cpus = len(cpu_handles)
        except AttributeError:
            cpu_handles = []
            num_cpus = len(psutil.cpu_percent(percpu=True))
        num_system_cpus = len(psutil.cpu_percent(percpu=True))
        self.pid = os.getpid()
        self.uid = os.getuid()
        self.slurm_job = os.environ.get("SLURM_JOB_ID", 0)
        self.levels = get_available_levels()
        self.process_pids = set()

        memory_limits = {
            level: detect_memory_limit(level, self.uid, self.slurm_job)
            for level in self.levels
        }

        self._process_backend = PsutilProcessBackend(
            pid=self.pid,
            process=self.process,
            uid=self.uid,
            slurm_job=self.slurm_job,
        )
        self._process_backend.setup()

        node_info = NodeInfo(
            node="local",
            num_cpus=num_cpus,
            num_system_cpus=num_system_cpus,
            num_gpus=0,  # updated after GPU discovery below
            gpu_memory=0.0,
            gpu_name="",
            memory_limits=memory_limits,
            cpu_handles=cpu_handles,
        )

        available = {
            "node_info": node_info,
            "uid": self.uid,
            "slurm_job": self.slurm_job,
        }

        # Build pipeline from config — collectors.yaml is the single source of truth
        cfg = load_collectors_config()
        self._pipeline = []
        num_gpus, gpu_memory, gpu_name = 0, 0.0, ""
        for collector_cfg in cfg["collectors"].values():
            collector_cfg = dict(collector_cfg)
            storage_cfg = collector_cfg.pop("storage")
            backend = instantiate(collector_cfg, **available)
            meta = backend.setup() or {}
            if "num_gpus" in meta:
                num_gpus = meta["num_gpus"]
                gpu_memory = meta.get("gpu_memory", 0.0)
                gpu_name = meta.get("gpu_name", "")
            self._pipeline.append((backend, make_handler(storage_cfg)))

        # Rebuild node_info with discovered GPU data
        node_info = NodeInfo(
            node="local",
            num_cpus=num_cpus,
            num_system_cpus=num_system_cpus,
            num_gpus=num_gpus,
            gpu_memory=gpu_memory,
            gpu_name=gpu_name,
            memory_limits=memory_limits,
            cpu_handles=cpu_handles,
        )
        # Propagate updated node_info to backends that declare it
        for backend, _ in self._pipeline:
            if hasattr(backend, "_node_info"):
                backend._node_info = node_info

        self.nodes = NodeDataStore()
        self.nodes.register_node(node_info)

        # Bootstrap: warm up process snapshots and IO counter state,
        # then derive per-level column names for schema pre-population.
        bootstrap_context: CollectionContext = {
            "process_pids": set(),
            "user_pids": set(),
            "slurm_pids": set(),
            "cpu": {},
            "rss": {},
            "io": {},
        }
        for backend, _ in self._pipeline:
            backend.snapshot(bootstrap_context)
        columns_by_level: dict[str, list[str]] = {}
        for level in self.levels:
            row: dict = {"time": 0.0}
            for collector, handler in self._pipeline:
                try:
                    row.update(handler.transform(collector.collect(level, bootstrap_context), level))
                except Exception:
                    pass
            columns_by_level[level] = list(row.keys())
        self.nodes.init_node_schema("local", columns_by_level)

        # session state
        self.is_imported = False
        self.session_source = None

    def _validate_level(self, level):
        if level not in self.levels:
            raise ValueError(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.INVALID_LEVEL
                ].format(level=level, levels=self.levels)
            )

    def _collect_metrics(self):
        """Collect one sample per level; return a list of flat dicts."""
        context: CollectionContext = {
            "process_pids": self.process_pids,
            "user_pids": set(),
            "slurm_pids": set(),
            "cpu": {},
            "rss": {},
            "io": {},
        }
        self._process_backend.snapshot(context)
        for backend, _ in self._pipeline:
            backend.snapshot(context)

        time_mark = time.perf_counter()
        rows = []
        for level in self.levels:
            row: dict = {"time": time_mark}
            for collector, handler in self._pipeline:
                row.update(handler.transform(collector.collect(level, context), level))
            rows.append(row)
        return rows

    def _collect_data(self):
        """Collect metrics at a fixed cadence anchored to an absolute timeline.

        Uses ``threading.Event.wait`` instead of ``time.sleep`` so that:
        * each tick is scheduled relative to a fixed epoch, preventing
          per-iteration sleep-overshoot from accumulating into drift;
        * ``stop()`` can signal the event and wake the thread instantly
          instead of blocking up to one full interval on ``thread.join``.

        The GIL is released during ``Event.wait`` just like ``time.sleep``.
        """
        next_tick = time.perf_counter()
        while not self._stop_event.is_set():
            self.process_pids = self._process_backend.get_process_pids()
            rows = self._collect_metrics()
            for level, row in zip(self.levels, rows):
                self.nodes.add_sample("local", level, row)
            self.n_measurements += 1

            next_tick += self.interval
            delay = next_tick - time.perf_counter()
            if delay > 0:
                self._stop_event.wait(delay)
            else:
                self.n_missed_measurements += 1
                next_tick = time.perf_counter()

    def start(self, interval: float = 1.0):
        if self.running:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.MONITOR_ALREADY_RUNNING]
            )
            return
        self.interval = interval
        self.start_time = time.perf_counter()
        self.wallclock_start_time = time.time()
        self.running = True
        self._stop_event = threading.Event()
        self.monitor_thread = threading.Thread(
            target=self._collect_data, daemon=True
        )
        self.monitor_thread.start()
        logger.info(
            EXTENSION_INFO_MESSAGES[ExtensionInfoCode.MONITOR_STARTED].format(
                pid=self.pid,
                interval=self.interval,
            )
        )

    def stop(self):
        self.running = False
        if hasattr(self, "_stop_event"):
            self._stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        self.stop_time = time.perf_counter()
        self.wallclock_stop_time = time.time()

        elapsed = self.stop_time - self.start_time
        expected = int(elapsed / self.interval) if self.interval > 0 else 0
        self.n_missed_measurements = max(0, expected - self.n_measurements)

        logger.info(
            EXTENSION_INFO_MESSAGES[ExtensionInfoCode.MONITOR_STOPPED].format(
                seconds=elapsed
            )
        )
        if self.n_measurements > 0:
            logger.info(
                EXTENSION_INFO_MESSAGES[ExtensionInfoCode.MISSED_MEASUREMENTS].format(
                    perc_missed_measurements=(
                        self.n_missed_measurements / expected
                        if expected > 0 else 0
                    )
                )
            )
