# Performance Monitoring

This package provides the performance-monitoring infrastructure for the
JUmPER IPython extension.  It defines a common **protocol** that all
monitors implement, several **concrete backends**, and the low-level
**metric collectors** (CPU, memory, GPU, I/O).

## Directory layout

```
monitor/
├── common.py                       # MonitorProtocol + utility monitors
├── metrics/                        # Pluggable metric collectors (psutil, NVML, …)
│   ├── cpu/                        #   Used by the Python-based monitors only
│   ├── gpu/                        #   (thread, subprocess_python, slurm_multinode).
│   ├── io/                         #   The C collector (native_c) reads /proc and
│   ├── memory/                     #   loads NVML directly — it has no dependency
│   └── process/                    #   on anything in metrics/.
└── backends/
    ├── thread/                     # In-process threaded monitor
    │   └── monitor.py              # PerformanceMonitor
    ├── subprocess_python/          # Out-of-process Python collector
    │   ├── _collector.py           # Python collector (run in child process)
    │   └── monitor.py              # SubprocessPerformanceMonitor
    ├── native_c/                   # Native C collector monitor
    │   ├── collector.c / Makefile  # C collector source & build
    │   └── monitor.py              # CSubprocessPerformanceMonitor
    └── slurm_multinode/            # Multi-node SLURM monitor
        ├── _collector.py           # Per-node collector (run via srun)
        ├── _log_writer.py          # JSON-Lines log writer
        ├── _node_discovery.py      # SLURM node list expansion
        └── monitor.py              # SlurmMultinodeMonitor
```

## MonitorProtocol

Every monitor exposes the same interface (`MonitorProtocol` in
`common.py`):

| Attribute / Method       | Description                              |
|--------------------------|------------------------------------------|
| `start(interval)`       | Begin collecting metrics                  |
| `stop()`                | Stop collecting and finalise timestamps   |
| `running`               | Whether the monitor is currently active   |
| `data`                  | `PerformanceData` container with results  |
| `interval`              | Sampling interval in seconds              |
| `num_cpus`              | Number of CPUs visible to the process     |
| `num_gpus`              | Number of GPUs detected                   |
| `memory_limits`         | Per-level memory limits (GiB)             |

The visualizer, reporter, and session exporter all program against this
protocol, so any backend can be swapped in transparently.

## Available monitors

### 1. Thread monitor (`"thread"`) (*deprecated, measurement resolution depends on GIL, potentially enough for non-CPU bound applications*)

```python
from jumper_extension.monitor.backends.thread import PerformanceMonitor
```

The original monitor.  Collects metrics in a daemon thread inside the
same Python process using **psutil** and **pynvml**.  Simple and
portable, but the GIL can delay sampling when the main thread is
CPU-bound.

### 2. Subprocess monitor — Python collector (`"default"`)

```python
from jumper_extension.monitor.backends.subprocess_python import SubprocessPerformanceMonitor
```

Spawns a **child Python process** that runs the same psutil-based
collection loop.  Results stream back to the parent over a pipe as
JSON lines.  Because collection happens in a separate process, it is
immune to GIL contention.

### 3. Native C collector monitor (`"native_c"`) (*experimental, useful for very high resolution, e.g. > 10Hz*)

```python
from jumper_extension.monitor.backends.native_c import CSubprocessPerformanceMonitor
```

Launches a **compiled C binary** (`jumper_collector`) that reads `/proc`
directly and speaks the same JSON-lines protocol.  Benefits:

- No Python startup overhead
- Minimal per-tick latency
- NVIDIA GPU metrics via dynamic loading of `libnvidia-ml.so`
  (no compile-time dependency; graceful fallback if absent)
- SLURM level auto-detected from the target process's environment

**Build the binary** before first use:

```bash
make -C jumper_extension/monitor/backends/native_c/
```

### 4. SLURM multi-node monitor (`"slurm_multinode"`) (*experimental*)

```python
from jumper_extension.monitor.backends.slurm_multinode import SlurmMultinodeMonitor
```

Discovers all nodes allocated to the current SLURM job, launches a
collector on each via `srun`, and aggregates their JSON sample streams into
a log file.  Designed for distributed HPC workloads.

## Selecting a monitor

From the IPython magic:

```
%perfmonitor_start --monitor default          # Python subprocess (default)
%perfmonitor_start --monitor native_c          # native C collector
%perfmonitor_start --monitor thread           # in-process thread
%perfmonitor_start --monitor slurm_multinode  # multi-node SLURM
```

Or programmatically via the service factory:

```python
service.start_monitoring(monitor_type="native_c")
```
