---
title: Jupyter API
---

# Jupyter API

The Jupyter‑level API is built around IPython line magics that wrap the underlying Python service. These magics are registered when you load the extension and are implemented by `PerfmonitorMagics` in combination with `PerfmonitorMagicAdapter`.

## Loading the extension

Enable the extension once per IPython or Jupyter session:

```python
%load_ext jumper_extension
```

This binds the `%perfmonitor_*` commands and related helpers to the current shell.

## Core workflow commands

The most common commands mirror the quickstart workflow:

- `%perfmonitor_start [interval]`  
  Start performance monitoring, optionally specifying the sampling interval in seconds (default: `1.0`). This forwards to `PerfmonitorService.start_monitoring`.

- `%perfmonitor_stop`  
  Stop the active monitoring session. Wraps `PerfmonitorService.stop_monitoring`.

- `%perfmonitor_perfreport [--cell RANGE] [--level LEVEL] [--text]`  
  Display a performance report for all or selected cells. Internally uses `PerfmonitorService.show_perfreport`.  
  - `--cell` accepts ranges like `5`, `2:8`, `:5`, or `3:`.  
  - `--level` selects the monitoring scope (`process`, `user`, `system`, `slurm`).  
  - `--text` switches from HTML to plain‑text output.

- `%perfmonitor_plot [--metrics LIST] [--cell RANGE] [--level LEVEL] [--save-jpeg FILE] [--pickle FILE]`  
  Open an interactive plot or produce direct plots and exports on top of the `PerformanceVisualizer`. The command delegates to `PerfmonitorService.plot_performance` and the underlying visualizer.

- `%perfmonitor_enable_perfreports [--level LEVEL] [--interval INTERVAL] [--text]`  
  Enable automatic reports after each cell. This configures the monitoring settings and calls `PerfmonitorService.enable_perfreports`.

- `%perfmonitor_disable_perfreports`  
  Disable automatic post‑cell reports, mapping to `PerfmonitorService.disable_perfreports`.

## Data export and import

You can export collected metrics and cell history for external analysis or later reuse:

- `%perfmonitor_export_perfdata [--file FILE] [--level LEVEL]`  
  Export performance data via `PerfmonitorService.export_perfdata`.  
  - Without `--file`, the command pushes a pandas `DataFrame` named according to `settings.export_vars.perfdata` into the user namespace.  
  - With `--file`, data is written to CSV/JSON using the monitor’s data adapter.

- `%perfmonitor_load_perfdata --file FILE`  
  Load previously exported performance data through `PerfmonitorService.load_perfdata`. The loaded `DataFrame` is pushed under `settings.loaded_vars.perfdata`.

- `%perfmonitor_export_cell_history [--file FILE]`  
  Export the executed cell history (`CellHistory`) using `PerfmonitorService.export_cell_history`. Without `--file`, a `DataFrame` named `cell_history_df` is pushed to the namespace.

- `%perfmonitor_load_cell_history --file FILE`  
  Load cell history from disk and attach it to the service via `PerfmonitorService.load_cell_history`.

## Session, resources, and helpers

Additional magics provide convenience and workflow features:

- `%perfmonitor_resources`  
  Show CPU, memory, and GPU availability for the current or imported session, implemented by `PerfmonitorService.show_resources`.

- `%cell_history` / `%show_cell_history`  
  Display an interactive table of all executed cells using the `CellHistory.show_itable` helper.

- `%export_session [target|target.zip]` and `%import_session <dir-or-zip>`  
  Export and import full monitoring sessions (performance data plus cell history) through `SessionExporter` and `SessionImporter`, coordinated by `PerfmonitorService.export_session` and `.import_session`.

- `%perfmonitor_fast_setup`  
  Run a one‑command setup that enables `ipympl`, starts monitoring, and turns on automatic reports. This combines IPython configuration with `PerfmonitorService.fast_setup`.

- `%start_write_script [output_path]` and `%end_write_script`  
  Record code from subsequent cells into a Python script via the `NotebookScriptWriter` managed by `PerfmonitorService.start_script_recording` and `.stop_script_recording`.

For a complete list of commands and brief descriptions, run:

```python
%perfmonitor_help
```

Details on the underlying Python methods are provided in the [Python API](python.md) section.

