---
title: String Based API
---

# String Based API

The string‑based API is implemented by `PerfmonitorMagicAdapter` in `jumper_extension.core.service`. It provides a thin layer that:

- Accepts command‑line style strings (as used by IPython magics).
- Parses arguments using `ArgParsers` built from the core parser utilities.
- Delegates the resulting structured options to methods on `PerfmonitorService`.

This layer is also used by the notebook script writer when generating reproducible monitoring scripts.

## **Role in the public API**

At runtime, the flow of a typical command is:

1. IPython calls the appropriate method on `PerfmonitorMagics` (for example `perfmonitor_perfreport`).
2. `PerfmonitorMagics` forwards the raw `line` string to the corresponding method on `PerfmonitorMagicAdapter`.
3. `PerfmonitorMagicAdapter` parses the string with the relevant parser (for example `parsers.perfreport`) and calls the corresponding method on `PerfmonitorService`.

This design keeps the parsing and text handling logic separate from the core monitoring and analysis code.

## **Example usage**

While you typically interact with the string‑based API indirectly through magics, you can also construct it directly:

```python
from jumper_extension.core.service import build_perfmonitor_magic_adapter

adapter = build_perfmonitor_magic_adapter()
adapter.perfmonitor_start("1.0")
adapter.perfmonitor_perfreport("--cell 2:5 --level user")
adapter.perfmonitor_export_perfdata("--file perf.csv --level system")
```

The commands are identical to the magic syntax but passed as plain strings. This is the same interface that `NotebookScriptWriter` relies on when emitting code for `monitored_script.py`.
