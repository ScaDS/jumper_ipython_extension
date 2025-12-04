---
title: Python API
---

# Python API

The Python API is centered around `PerfmonitorService`, a standalone orchestration class defined in `jumper_extension.core.service`. It wires together monitoring, visualization, reporting, cell history, and session management and can be used directly from Python code without IPython magics.

## Constructing a service

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

## Core service methods

Key methods of `PerfmonitorService` include:

- `start_monitoring(interval: Optional[float] = None) -> Optional[ExtensionErrorCode]`  
  Start the monitor with a given sampling interval. If `interval` is `None`, the default from `settings.monitoring.default_interval` is used. When an offline session is currently attached, a new live `PerformanceMonitor` is created.

- `stop_monitoring() -> None`  
  Stop the active monitoring session and mark `settings.monitoring.running` as `False`.

- `show_perfreport(cell_range: Optional[Tuple[int, int]] = None, level: Optional[str] = None, text: bool = False) -> None`  
  Display a performance report using the `PerformanceReporter`.  
  - `cell_range` restricts the report to specific cells.  
  - `level` selects the monitoring scope.  
  - `text` switches between HTML and plain‑text rendering.

- `plot_performance() -> None`  
  Open an interactive visualization using the attached `PerformanceVisualizer`. Works with both live and imported sessions.

- `enable_perfreports(level: str, interval: Optional[float] = None, text: bool = False) -> None`  
  Enable automatic reports after each cell. This updates `settings.perfreports` and calls `start_monitoring` if necessary.

- `disable_perfreports() -> None`  
  Disable automatic cell‑level reports.

## Data access and export

The service provides helpers to access collected data as pandas `DataFrame` objects or export them to disk:

- `export_perfdata(file: Optional[str] = None, level: Optional[str] = None) -> Dict[str, pd.DataFrame]`  
  - If `file` is `None`, returns a dictionary mapping `settings.export_vars.perfdata` to a `DataFrame` produced by `monitor.data.view`.  
  - If `file` is set, forwards to `monitor.data.export` and returns an empty dictionary.

- `load_perfdata(file: str) -> Dict[str, pd.DataFrame]`  
  Load performance data from a file via `monitor.data.load` and return it under the key `settings.loaded_vars.perfdata`.

- `export_cell_history(file: Optional[str] = None) -> Dict[str, pd.DataFrame]`  
  - Without `file`, return a dictionary mapping `settings.export_vars.cell_history` to the `CellHistory.view()` `DataFrame`.  
  - With `file`, export cell history via `CellHistory.export`.

- `load_cell_history(file: str) -> Dict[str, pd.DataFrame]`  
  Load cell history using `CellHistory.load` and return it under `settings.loaded_vars.cell_history`.

## Sessions, scripts, and utilities

Additional service methods cover higher‑level workflows:

- `show_resources() -> None`  
  Print information about CPUs, memory, and GPUs available to the current or imported session. When metrics come from an imported session, the source is included in the message.

- `show_cell_history() -> None`  
  Show an interactive table of executed cells using `CellHistory.show_itable`.

- `export_session(path: Optional[str] = None) -> None`  
  Export a full session (performance data and cell history) through `SessionExporter`. When `path` ends with `.zip`, a zip archive is created automatically.

- `import_session(path: str) -> None`  
  Import a session via `SessionImporter` and attach it to the service. On success, a log message announces the source of the imported session.

- `fast_setup() -> None`  
  Convenience method that starts monitoring with a `1.0` second interval and enables HTML per‑cell performance reports at the `process` level.

- `start_script_recording(output_path: Optional[str] = None) -> None` and `stop_script_recording() -> Optional[str]`  
  Coordinate `NotebookScriptWriter` to record cell code and write it to a Python script. When `output_path` is `None`, a filename is generated automatically.

- `monitored()` (context manager)  
  Provide a Python `with`‑statement context that marks the enclosed block as a virtual cell and ensures that pre‑ and post‑cell hooks are invoked around it.

For direct interaction with string‑based commands or IPython magics, see the [String Based API](string.md) and [Jupyter API](jupyter.md) sections.

## API reference

The following reference is generated directly from the Python source code using mkdocstrings.

### `PerfmonitorService`

::: jumper_extension.core.service.PerfmonitorService

### Builder functions

::: jumper_extension.core.service.build_perfmonitor_service

::: jumper_extension.core.service.build_perfmonitor_magic_adapter

