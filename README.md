[![Unit Tests](https://github.com/ScaDS/jumper_ipython_extension/actions/workflows/test.yml/badge.svg)](https://github.com/ScaDS/jumper_ipython_extension/actions/workflows/test.yml)
[![Formatting](https://github.com/ScaDS/jumper_ipython_extension/actions/workflows/formatter.yml/badge.svg)](https://github.com/ScaDS/jumper_ipython_extension/actions/workflows/formatter.yml)
[![Static Analysis](https://github.com/ScaDS/jumper_ipython_extension/actions/workflows/linter.yml/badge.svg)](https://github.com/ScaDS/jumper_ipython_extension/actions/workflows/linter.yml)

<p align="center">
<img width="450" src="doc/JUmPER01.png"/>
</p>

# JUmPER: Jupyter meets Performance

JUmPER brings performance engineering to Jupyter. It consists of the two repositories:

- JUmPER Ipython extension (this repository)

This extension is for real-time performance monitoring in IPython environments and Jupyter notebooks. It allows you to gather performance data on CPU usage, memory consumption, GPU utilization, and I/O operations for individual cells and present it in the notebook/IPython session either as text report or as a plot.

- Score-P Jupyter kernel Python (https://github.com/score-p/scorep_jupyter_kernel_python)

The Score-P kernel allows you to instrument, and trace or profile your Python code in Jupyter using [Score-P](https://score-p.org/) for in-detail performance analysis tasks. The Score-P kernel and the IPython extension can be seamlessly integrated.


# Table of Content

* [Installation](#installation)
* [Quick Start](#quick-start)
	+ [Load the Extension](#load-the-extension)
	+ [Basic Usage](#basic-usage)
* [Metrics Collection](#metrics-collection)
	+ [Performance Monitoring Levels](#performance-monitoring-levels)
	+ [Collected Metrics](#collected-metrics)
* [Available Commands](#available-commands)
* [Contribution and Citing](#contribution-and-citing)

## Installation

```bash
pip install jumper_extension
```

or install it from source:

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
   %perfmonitor_perfreport
   %perfmonitor_perfreport --cell 2:5 --level user
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

   Options:
   - `--cell RANGE`: Specify cell range (e.g., `5`, `2:8`, `:5`)
   - `--level LEVEL`: Choose monitoring level (`process`, `user`, `system`, `slurm`)

4. **Plot performance data**:
   ```python
   %perfmonitor_plot
   ```

   Opens an interactive plot with widgets to explore performance metrics over time, filter by cell ranges, and select different monitoring levels.

![](doc/plot_out.png)



5. **View cell execution history**:
   ```python
   %cell_history
   ```

   Shows an interactive table of all executed cells with timestamps and durations.

6. **Stop monitoring**:
   ```python
   %perfmonitor_stop
   ```

7. **Export data for external analysis**:
   ```python
   %perfmonitor_export_perfdata my_performance.csv --level system
   %perfmonitor_export_cell_history my_cells.json
   ```
   Export performance measurements for entire notebook and cell execution history with timestamps, allowing you to project measurements onto specific cells.

## Metrics Collection

### Performance Monitoring Levels

The extension supports four different levels of metric collection, each providing different scopes of system monitoring:

- **Process**: Metrics for the current Python process only
- **User**: Metrics for all processes belonging to the current user
- **System**: System-wide metrics across all processes and users (if visible)
- **Slurm**: Metrics for processes within the current SLURM job

### Collected Metrics

| Metric | Description |
|--------|-------------|
| `cpu_util` | CPU utilization percentage |
| `memory` | Memory usage in GB |
| `io_read_count` | Total number of read I/O operations |
| `io_write_count` | Total number of write I/O operations |
| `io_read_mb` | Total data read in MB |
| `gpu_util` | GPU compute utilization percentage across GPUs |
| `gpu_band` | GPU memory bandwidth utilization percentage |
| `gpu_mem` | GPU memory usage in GB across GPUs |
| `io_write_mb` | Total data written in MB |

*Note: GPU metrics require NVIDIA GPUs with pynvml library. Memory limits are automatically detected from SLURM cgroups when available.*

## Available Commands

| Command | Description |
|---------|-------------|
| `%perfmonitor_fast_setup` | Fast setup of JUmPER. Starts monitor (1.0s interval), enables perfreports (--level process) and interactive plots (ipympl) |
| `%perfmonitor_help` | Show all available commands with examples |
| `%perfmonitor_resources` | Display available hardware resources |
| `%perfmonitor_start [interval]` | Start monitoring (default: 1 second interval) |
| `%perfmonitor_stop` | Stop monitoring |
| `%perfmonitor_perfreport [--cell RANGE] [--level LEVEL]` | Show performance report for specific cell range and monitoring level |
| `%perfmonitor_plot` | Interactive plot with widgets for exploring performance data |
| `%cell_history` | Show execution history of all cells with interactive table |
| `%perfmonitor_enable_perfreports` | Auto-generate reports after each cell |
| `%perfmonitor_disable_perfreports` | Disable auto-reports |
| `%perfmonitor_export_perfdata [--file filename] [--level LEVEL]` | Export performance data to dataframe. Export performance data to CSV if `--file` is set. |
| `%perfmonitor_export_cell_history [filename]` | Export cell history to CSV/JSON |

## Contribution and Citing:
PRs are welcome. Feel free to use the pre-commit hooks provided in .githooks

If you publish some work using the kernel, we would appreciate if you cite one of the following papers:

```
Werner, E., Rygin, A., Gocht-Zech, A., Döbel, S., & Lieber, M. (2024, November).
JUmPER: Performance Data Monitoring, Instrumentation and Visualization for Jupyter Notebooks.
In SC24-W: Workshops of the International Conference for High Performance Computing, Networking, Storage and Analysis (pp. 2003-2011). IEEE.
https://www.doi.org/10.1109/SCW63240.2024.00250
```
