---
title: JUmPER API
---

# JUmPER API

The JUmPER IPython extension exposes a layered public API that lets you work with performance monitoring at different levels:

- The **Jupyter API** provides IPython line magics for interactive use in notebooks and shells.
- The **Python API** exposes a programmatic interface centered around `PerfmonitorService` for use in scripts and libraries.
- The **String Based API** offers a thin adapter between textual magic commands and the Python service. It is also used by the script writer when generating reproducible monitoring scripts.

The following subsections describe each of these layers and how they relate to each other.

For architectural context and component diagrams, see the [Architecture](../internals/architecture.md) page in the Internals section.

