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
### CPU
- Per-core utilization
- Average utilization across available cores

### Memory
- Total memory usage

### GPU (if available)
- GPU compute utilization
- GPU memory bandwidth utilization
- GPU memory usage

### I/O Operations
- Read/write operation counts
- Data transfer volume