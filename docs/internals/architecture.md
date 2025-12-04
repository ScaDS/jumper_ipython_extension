---
title: Architecture
---

# Architecture

The JUmPER IPython extension is organized as a set of layers that separate public APIs from internal monitoring, visualization, and storage components. The diagrams in the `Public API` and `Underlying Structure` images, together with the internal documentation, describe how these layers fit together.

## **Public API layers**

![Public API diagram](../img/Public%20API.jpg)

The `Public API` diagram shows how user‑facing entry points build on top of a single core service:

- **IPython.core.magic.Magics**  
  The base IPython API for defining line magics. The `PerfmonitorMagics` class subclasses this type.

- **PerfmonitorMagics**  
  A thin bridge between IPython and the monitoring stack. Each magic (such as `%perfmonitor_start` or `%perfmonitor_plot`) lives here and forwards calls to the adapter layer.

- **PerfmonitorMagicAdapter (string‑based API)**  
  An adapter that accepts raw command strings, parses them with argument parsers, and translates them into structured calls on `PerfmonitorService`. It also serves as the target for the notebook script writer when generating `monitored_script.py`.

- **PerfmonitorService (Python API)**  
  The central orchestration class with a standalone Python interface. It handles magic‑level operations, wires in adapters, and provides methods that can be called directly from Python code.

Together, these components allow the same core logic to be reused from notebooks, scripts, and direct Python APIs.

## **Underlying structure**

![Underlying structure diagram](../img/Underlying%20Structure.jpg)

The `Underlying Structure` diagram focuses on the internal levels that support the public API:

- **API level**  
  Entry points such as IPython, generated scripts (`monitored_script.py`), and direct Python API usage. All of them ultimately call into `PerfmonitorService`.

- **Service level**  
  The service coordinates configuration and delegates work to adapters. It is free of IPython‑specific concepts and can be used in non‑notebook contexts.

- **Top‑level adapters**  
  User‑facing features that operate on monitoring data:
  - `reporter` — builds text and HTML performance reports using templates.  
  - `visualizer` — renders interactive plots and dashboards on top of collected metrics.  
  - `session` — handles export/import of complete sessions, including performance data and cell history.  
  - `script_writer` — records cells and monitoring calls into reproducible Python scripts.

- **Low‑level adapters**  
  Components that directly interact with the runtime environment and raw data:
  - `monitor` — collects metrics such as CPU, memory, GPU, and I/O usage. Supports both live and offline modes.  
  - `data` — stores performance samples and exposes them as pandas `DataFrame` objects for higher‑level components.  
  - `cell_history` — records executed cells, timestamps, durations, and metadata, allowing reports and plots to be aligned with notebook cells.  
  - `analyzer` — performs higher‑level analysis and classification on top of recorded metrics.

These layers are wired together by the `jumper_extension.core` package, as summarized in `INTERNAL_DOCS.md`.

## **Package layout**

The directory layout of the main package is:

```text
jumper_extension/ — main Python package  
├─ __init__.py — package initialization and public exports  
├─ utilities.py — shared helper functions used across core and adapters  
├─ logging_config.py — logging configuration helpers for the extension  
├─ logo.py — ASCII/logo utilities for branding and display  
├─ core/ — stable domain model and orchestration (no feature-specific logic)  
│  ├─ __init__.py  
│  ├─ service.py — PerfmonitorService and PerfmonitorMagicAdapter (central orchestration)  
│  ├─ state.py — state management, settings dataclasses and snapshots of user configuration  
│  ├─ parsers.py — argument parsers for magics and CLI-like flows  
│  └─ messages.py — centralized user-facing messages and formatting  
├─ adapters/ — feature implementations and integrations  
│  ├─ __init__.py  
│  ├─ monitor.py — live and offline performance monitors (PerformanceMonitor, OfflinePerformanceMonitor)  
│  ├─ cell_history.py — capture and persistence of executed notebook cells  
│  ├─ data.py — performance data model and CSV/JSON import/export  
│  ├─ reporter.py — text/HTML performance reporting built on templates  
│  ├─ analyzer.py — performance classification logic  
│  ├─ visualizer.py — plotting and interactive visualization  
│  ├─ session.py — session export/import (CSV/JSON + manifest, offline monitor wiring)  
│  └─ script_writer.py — notebook-to-script recording based on cell history and settings snapshots  
├─ ipython/ — thin IPython integration layer  
│  ├─ __init__.py  
│  ├─ extension.py — load/unload hooks for the IPython extension  
│  ├─ magics.py — line magics delegating into PerfmonitorMagicAdapter  
│  └─ utilities.py — IPython-specific helper utilities  
└─ templates/ — frontend components (HTML templates, CSS, assets)  
   ├─ __init__.py  
   └─ report/ — performance report templates, styles, and static assets
```

For details on individual public methods and usage patterns, see the [Public API](../api/index.md) section.
