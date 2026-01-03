---
title: Quickstart
---

# Quickstart

This quickstart shows how to enable the JUmPER IPython extension in a notebook, start monitoring, and inspect basic performance data for a few cells.

All examples below assume that `jumper_extension` has been installed as described in the [Installation](installation.md) section.

## **Enable the extension**

In a Jupyter notebook or IPython shell, load the extension once per session:

```python
%load_ext jumper_extension
```

This registers the `%perfmonitor_*` magic commands and wires them into the underlying monitoring service.

## **Minimal monitoring workflow**

### 1. Start monitoring

Begin collecting performance data for subsequent cells:

```python
%perfmonitor_start [interval]
```

- `interval` is optional and specifies how often metrics are sampled in seconds.
- If omitted, the default interval of `1` second is used.

### 2. Run your code

Execute the cells you want to profile as usual. While monitoring is active, JUmPER records CPU, memory, GPU, and I/O metrics over time.

### 3. View a performance report

Show an aggregate report for the current session:

```python
%perfmonitor_perfreport
%perfmonitor_perfreport --cell 2:5 --level user
```

- Without arguments, the report covers all cells executed so far.
- `--cell RANGE` restricts the analysis to specific cells (for example `5`, `2:8`, `:5`, or `3:`).
- `--level LEVEL` selects the monitoring scope: `process`, `user`, `system`, or `slurm` (if available).

The report prints aggregated metrics such as CPU utilization, memory usage, GPU utilization, and GPU memory across the selected range.

### 4. Plot performance data

Open an interactive plot with widgets for exploring metrics over time:

```python
%perfmonitor_plot
```

The plot lets you:

- Zoom into interesting regions of the timeline.
- Filter by cell ranges.
- Switch between monitoring levels.

You can also use a direct, non-widget mode and export results:

```python
%perfmonitor_plot --metrics cpu_summary,memory
%perfmonitor_plot --metrics cpu_summary --level user --cell 2:5
%perfmonitor_plot --metrics cpu_summary,memory --save-jpeg performance_analysis.jpg
%perfmonitor_plot --metrics cpu_summary --level user --pickle analysis_data.pkl
```

- `--metrics` accepts a comma‑separated list of metric keys such as `cpu_summary`, `memory`, `io_read`, `io_write`, `gpu_util_summary`, `gpu_band_summary`, and `gpu_mem_summary`.
- `--save-jpeg` writes the current view to an image file.
- `--pickle` exports the plot data for later interactive analysis.

### 5. Inspect cell execution history

Review all executed cells with their timestamps and durations:

```python
%cell_history
```

This opens an interactive table that lets you correlate individual cells with collected performance metrics.

### 6. Stop monitoring

When you are done collecting data, stop the monitor:

```python
%perfmonitor_stop
```

## **One‑command fast setup**

For a fully configured environment with interactive plotting and automatic reports, use the fast setup command:

```python
%perfmonitor_fast_setup
```

This command:

- Enables `ipympl`‑based interactive plots in the current notebook.
- Starts the performance monitor with a `1.0` second interval.
- Enables automatic performance reports after each cell at the `process` level.

For detailed options and additional commands, see the [Public API](../api/index.md) section.
