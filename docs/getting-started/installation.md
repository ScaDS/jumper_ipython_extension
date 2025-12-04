---
title: Installation
---

# Installation

This extension is distributed as a standard Python package and can be installed from PyPI or from source.

## Install from PyPI

Install the latest released version into your current environment:

```bash
pip install jumper_extension
```

This installs the IPython extension, its monitoring dependencies, and the entry point that allows `%load_ext jumper_extension` to work inside IPython and Jupyter.

## Install from source

If you are working from a clone of the repository or want the very latest changes, install the package from the project root:

```bash
pip install .
```

This builds and installs the local copy of `jumper_extension` into your active environment.

## Optional GPU support

JUmPER can collect GPU metrics for both NVIDIA and AMD GPUs. GPU support is optional; CPU and memory monitoring work without it.

For NVIDIA GPUs, install:

```bash
pip install pynvml
```

For AMD GPUs, install:

```bash
pip install ADLXPybind
```

Both GPU libraries can be installed at the same time to monitor mixed GPU systems. JUmPER will automatically detect which backends are available.

## Environment configuration

JUmPER writes logs to a configurable directory. To change the default location, set the `JUMPER_LOG_DIR` environment variable before starting your notebook or IPython session:

```bash
export JUMPER_LOG_DIR=/path/to/logs
```

If this variable is not set, logs are stored in a directory under the user’s home folder.

