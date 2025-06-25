# JUmPER Extension

This is JUmPER IPython extension for real-time performance monitoring in IPython environments and Jupyter notebooks. It allows you to gather performance data on CPU usage, memory consumption, GPU utilization, and I/O operations for individual cells and present it in the notebook/IPython session either as text report or as a plot. The extension can be naturally integrated with [JUmPER Jupyter kernel](https://github.com/score-p/scorep_jupyter_kernel_python/) for most comprehensive analysis of notebook.

## Installation

```bash
pip install .
```

## Quick Start

### Load the Extension

```python
%load_ext jumper_extension
```

### Basic Usage

1. **Start monitoring**:
   ```python
   %perfmonitor_start [interval]
   ```

   `interval` is an optional argument for configuring frequency of performance data gathering (in seconds), set to 1 by default. This command launches a performance monitoring daemon.

2. **Run your code**

3. **View performance report**:
   ```python
   %perfmonitor_perfreport [cell]
   ```

   Will print aggregate performance report for entire notebook execution so far:

   ```
   ----------------------------------------
   Performance Report
   ----------------------------------------
   Duration: 11.08s
   Metric                    AVG      MIN      MAX      TOTAL   
   -----------------------------------------------------------------
   CPU Util (Across CPUs)    10.55    3.86     45.91    -       
   Memory (GB)               7.80     7.74     7.99     15.40   
   GPU Util (Across GPUs)    27.50    5.00     33.00    -       
   GPU Memory (GB)           0.25     0.23     0.32     4.00    
   ```

   Pass cell number to see only this cell performance report. Refer to `%cell_history` to identify it from notebook execution history.

4. **Plot performance data**:
   ```python
   %perfmonitor_plot [cell]
   ```

   Plot a more detailed overview of performance metrics over time.

5. **Stop monitoring**:
   ```python
   %perfmonitor_stop
   ```

6. ### Export data for external analysis
   ```python
   %perfmonitor_export_perfdata my_performance.csv
   %perfmonitor_export_cell_history my_cells.json
   ```
   Export performance measurements for entire notebook and cell execution history with timestamps, allowing you to project measurements onto specific cells.

## Available Commands

| Command | Description |
|---------|-------------|
| `%perfmonitor_help` | Show all available commands |
| `%perfmonitor_resources` | Display available hardware resources |
| `%perfmonitor_start [interval]` | Start monitoring (default: 1 second interval) |
| `%perfmonitor_stop` | Stop monitoring |
| `%perfmonitor_perfreport [cell]` | Show performance report for specific cell or latest |
| `%perfmonitor_plot [cell]` | Plot performance data for specific cell or all data |
| `%cell_history` | Show execution history of all cells |
| `%perfmonitor_enable_perfreports` | Auto-generate reports after each cell |
| `%perfmonitor_disable_perfreports` | Disable auto-reports |
| `%perfmonitor_export_perfdata [filename]` | Export performance data to CSV |
| `%perfmonitor_export_cell_history [filename]` | Export cell history to JSON |

## Monitored Metrics

The following table describes all metrics collected by the performance monitor:

| Metric | Description | Collection Method | Level |
|------------------|-------------|------------------|--------|
| `memory_usage_gb` | Total system memory usage in GB | `psutil.virtual_memory()` | System |
| `cpu_util` | CPU utilization across cores | `psutil.cpu_percent(percpu=True)` | System |
| `io_read_count` | Total number of read I/O operations | `psutil.Process().io_counters().read_count` | Process |
| `io_write_count` | Total number of write I/O operations | `psutil.Process().io_counters().write_count` | Process |
| `io_read_mb` | Total data read in MB | `psutil.Process().io_counters().read_bytes` | Process |
| `io_write_mb` | Total data written in MB | `psutil.Process().io_counters().write_bytes` | Process |
| `gpu_util` | GPU compute utilization across GPUs | `pynvml.nvmlDeviceGetUtilizationRates().gpu` | System |
| `gpu_band` | GPU memory bandwidth utilization across GPUs | `pynvml.nvmlDeviceGetUtilizationRates().memory` | System |
| `gpu_mem` | GPU memory usage in GB across GPUs | `pynvml.nvmlDeviceGetMemoryInfo()` | System |

### Collection Levels

- **System**: Metrics collected for the entire system across all users and processes
- **Process**: Metrics collected specifically for the current Python process
#- **User**: *(Future)* Metrics for all processes owned by the current user

### Notes

- GPU metrics are only available when NVIDIA drivers and `pynvml` library are installed
- Memory detection is SLURM-aware when running in SLURM environments
- CPU metrics are limited to cores available to the current process (respects CPU affinity)
- I/O metrics track only the main Python process, not child processes