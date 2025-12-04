---
title: Jupyter API
---

# Jupyter API

The Jupyter‑level API is built around IPython line magics that wrap the underlying Python service. These magics are registered when you load the extension and are implemented by `PerfmonitorMagics` in combination with `PerfmonitorMagicAdapter`.

## **Loading the extension**

Enable the extension once per IPython or Jupyter session:

```python
%load_ext jumper_extension
```

This binds the `%perfmonitor_*` commands and related helpers to the current shell.

## **Core workflow commands**

The most common magics mirror the quickstart workflow and control
monitoring, reporting, and plotting.

::: jumper_extension.ipython.magics.PerfmonitorMagics
    options:
      show_root_heading: false
      show_root_full_path: false
      show_root_toc_entry: false
      members:
        - perfmonitor_start
        - perfmonitor_stop
        - perfmonitor_perfreport
        - perfmonitor_plot
        - perfmonitor_enable_perfreports
        - perfmonitor_disable_perfreports

## **Data export and import**

You can export collected metrics and cell history for external analysis
or later reuse.

::: jumper_extension.ipython.magics.PerfmonitorMagics
    options:
      show_root_heading: false
      show_root_full_path: false
      show_root_toc_entry: false
      members:
        - perfmonitor_export_perfdata
        - perfmonitor_load_perfdata
        - perfmonitor_export_cell_history
        - perfmonitor_load_cell_history

## **Session, resources, and helpers**

Additional magics provide resources overview, session management, and
script recording helpers.

::: jumper_extension.ipython.magics.PerfmonitorMagics
    options:
      show_root_heading: false
      show_root_full_path: false
      show_root_toc_entry: false
      members:
        - perfmonitor_resources
        - show_cell_history
        - export_session
        - import_session
        - perfmonitor_fast_setup
        - start_write_script
        - end_write_script

For a complete list of commands and brief descriptions, run:

```python
%perfmonitor_help
```

Details on the underlying Python methods are provided in the [Python API](python.md) section.
