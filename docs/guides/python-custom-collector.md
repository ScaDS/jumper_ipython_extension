# Custom Python Collectors

A **collector** lets you add a new metric group to the `thread` or
`subprocess_python` monitors without touching the monitor itself — register a
`CollectorBackend` + `StorageHandler` pair in `default.yaml` and it is loaded
automatically.

!!! note
    This guide covers the **Python** monitor backends (`thread`, `subprocess_python`).
    For the native C binary see [Custom C Collectors](c-custom-collector.md). For an
    overview of the config system this `default.yaml` is part of, see
    [Configuration](configuration.md).

---

## Step 1 — Create the collector module

By convention each metric lives in its own subdirectory under
`jumper_extension/monitor/metrics/`:

```
jumper_extension/monitor/metrics/
└── your_metric/
    ├── common.py    # YourCollectorBackend — ABC with typed collect() signature
    ├── psutil.py    # PsutilYourCollector  — concrete implementation
    └── __init__.py  # re-exports YourCollectorBackend
```

### Example — `NetworkCollector`

**`metrics/network/common.py`**
```python
from abc import abstractmethod
from jumper_extension.monitor.metrics.common import CollectorBackend
from jumper_extension.monitor.metrics.context import CollectionContext

class NetworkCollectorBackend(CollectorBackend):
    name = "network-base"

    @abstractmethod
    def collect(self, level: str, context: CollectionContext) -> list[int]: ...
```

**`metrics/network/psutil.py`**
```python
import psutil
from jumper_extension.monitor.metrics.context import CollectionContext
from jumper_extension.monitor.metrics.network.common import NetworkCollectorBackend

class PsutilNetworkCollector(NetworkCollectorBackend):
    """System-wide network I/O via psutil.

    psutil does not expose per-process network counters on Linux without root,
    so only the 'system' level returns real data; other levels report zeros.
    """
    name = "network-psutil"

    def collect(self, level: str, context: CollectionContext) -> list[int]:
        if level == "system":
            net = psutil.net_io_counters()
            if net:
                return [net.bytes_sent, net.bytes_recv,
                        net.packets_sent, net.packets_recv]
        return [0, 0, 0, 0]
```

**`metrics/network/__init__.py`**
```python
from jumper_extension.monitor.metrics.network.common import NetworkCollectorBackend
```

---

## Step 2 — Pick a StorageHandler

A handler converts the value returned by `collect()` into DataFrame columns.
Built-in handlers live in `jumper_extension/monitor/metrics/handlers.py`:

| Handler | Raw type | Output columns |
|---------|----------|----------------|
| `ScalarHandler(column="x")` | `float` | `{"x": v}` |
| `PerDeviceAggregateHandler(prefix="p_")` | `list[float]` | `p_0, p_1, …, p_avg, p_min, p_max` |
| `PerDeviceMultiAggregateHandler(prefix="p_", metrics=[…])` | `tuple[list[float], …]` | fan-out of PerDeviceAggregate per metric |
| `CumulativeRateHandler(columns=[…])` | `list[int]` | per-column delta/second rates |
| `NoOpHandler()` | `None` | `{}` |

`NetworkCollector` returns cumulative byte/packet counters → pair it with
`CumulativeRateHandler` to get rates automatically.

---

## Step 3 — Register in `default.yaml`

```yaml
# jumper_extension/config/collectors/python/default.yaml
collectors:
  # ... existing collectors ...

  network:
    _target_: jumper_extension.monitor.metrics.network.psutil.PsutilNetworkCollector
    inject: []
    handler:
      _target_: jumper_extension.monitor.metrics.handlers.CumulativeRateHandler
      columns: [net_bytes_sent, net_bytes_recv, net_packets_sent, net_packets_recv]
```

### The `inject:` key

`inject:` lists `PerformanceMonitor` attributes passed as constructor arguments:

| Value | Type | What you get |
|-------|------|-------------|
| `[]` | — | no injected dependencies |
| `[node_info]` | `NodeInfo` | CPU/GPU count, memory limits, CPU handles |
| `[uid, slurm_job]` | `int, str\|int` | current user ID and SLURM job ID |
| `[pid, process, uid, slurm_job]` | mixed | full process context (used by built-in process collector) |

---

With the collector registered, its columns are available in
`%perfmonitor_plot --metrics`. To surface them in the default plot widget see
[Visualizing Custom Collector Metrics](visualizing-custom-collector-metrics.md).
