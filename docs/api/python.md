---
title: Python API
---

# Python API

The Python API is centered around `PerfmonitorService`, a standalone orchestration class defined in `jumper_extension.core.service`. It wires together monitoring, visualization, reporting, cell history, and session management and can be used directly from Python code without IPython magics.

## **Constructing a service**

The recommended entry point is the factory function `build_perfmonitor_service`:

```python
from jumper_extension.core.service import build_perfmonitor_service

service = build_perfmonitor_service()
```

This function creates:

- A `Settings` instance holding monitoring and reporting configuration.
- A `PerformanceMonitor` for collecting metrics.
- A `CellHistory` tracker for executed cells.
- A `PerformanceVisualizer` and `PerformanceReporter` attached to the monitor and cell history.
- A `NotebookScriptWriter` for script recording.

The returned `PerfmonitorService` exposes methods that mirror the high‑level commands used by the Jupyter API.

::: jumper_extension.core.service
    options:
      show_root_heading: false
      show_root_full_path: false
      show_root_toc_entry: false
      members:
        - build_perfmonitor_service
        - build_perfmonitor_magic_adapter

## **Core service methods**

Core methods control monitoring, plotting, and automatic per‑cell reports.

::: jumper_extension.core.service.PerfmonitorService
    options:
      show_root_heading: false
      show_root_full_path: false
      show_root_toc_entry: false
      members:
        - start_monitoring
        - stop_monitoring
        - enable_perfreports
        - disable_perfreports
        - show_perfreport
        - plot_performance

## **Data access and export**

The service exposes helpers for accessing collected data as pandas
`DataFrame` objects and exporting or loading them from disk.

::: jumper_extension.core.service.PerfmonitorService
    options:
      show_root_heading: false
      show_root_full_path: false
      show_root_toc_entry: false
      members:
        - export_perfdata
        - load_perfdata
        - export_cell_history
        - load_cell_history

## **Sessions, scripts, and utilities**

For higher‑level workflows, the service also exposes helpers for
resources, sessions, and script recording.

For direct interaction with string‑based commands or IPython magics,
see the [String Based API](string.md) and [Jupyter API](jupyter.md)
sections.

::: jumper_extension.core.service.PerfmonitorService
    options:
      show_root_heading: false
      show_root_full_path: false
      show_root_toc_entry: false
      members:
        - show_resources
        - show_cell_history
        - export_session
        - import_session
        - fast_setup
        - start_script_recording
        - stop_script_recording
        - monitored
