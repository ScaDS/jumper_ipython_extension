---
title: Core
---

# Core

The core package (`jumper_extension.core`) defines the domain model, configuration, and parsing logic that underpin the JUmPER IPython extension. For the public Python API of `PerfmonitorService`, see the [Python API](../api/python.md) section; the content below focuses on supporting modules and helper functions and is generated directly from the code.

## **State**

The `state` module contains dataclasses describing runtime settings for
monitoring, automatic reports, and exported or loaded variables.

::: jumper_extension.core.state

## **Parsers**

::: jumper_extension.core.parsers

## **Messages**

The `messages` module defines error and info codes together with their
human-readable message templates.

::: jumper_extension.core.messages

## **Service helpers**

The service itself is documented in the public API; this section exposes the helper functions that construct it.

::: jumper_extension.core.service
    options:
      show_root_heading: false
      show_root_full_path: false
      show_root_toc_entry: false
      members:
        - build_perfmonitor_service
        - build_perfmonitor_magic_adapter
