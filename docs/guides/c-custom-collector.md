# Custom C Collectors

The `native_c` monitor is a compiled binary that reads `/proc` directly.  Its
metric backends are C translation units — not Python classes — and are registered
at compile time, not via `collectors.yaml`.

!!! note
    For the **Python** monitor backends (`thread`, `subprocess_python`) see
    [Custom Python Collectors](python-custom-collector.md).

---

## Source layout

```
native_c/
├── monitor.c / monitor.h   # main loop, shared types, registry
├── Makefile
└── metrics/
    ├── cpu/    cpu.c / cpu.h
    ├── memory/ memory.c / memory.h
    ├── io/     io.c / io.h
    └── gpu/    gpu.c / gpu.h
```

Each backend exposes a single `const CCollector <name>_collector` symbol and
includes `monitor.h` (available via the `-I$(DIR)` compile flag).

---

## `CCollector` interface

```c
typedef struct {
    const char *name;
    void (*setup)(void);
    void (*snapshot)(TickContext *ctx);
    void (*emit_columns)(FILE *fp, int level_idx, int n_cpus, int n_gpus);
    void (*emit_sample)(FILE *fp, int level_idx, TickContext *ctx, double dt);
    void (*post_tick)(void);
    void (*teardown)(void);
} CCollector;
```

| Callback | When called | Set to `NULL` if |
|---|---|---|
| `setup` | once, before main loop | no one-time init needed |
| `snapshot` | every tick, before any emit | no per-tick `/proc` reads needed |
| `emit_columns` | once per level at start (ready handshake) | backend adds no columns |
| `emit_sample` | every tick, per level | backend adds no columns |
| `post_tick` | after all levels emitted | no post-emit cache work needed |
| `teardown` | on clean shutdown | no resources to release |

`TickContext` carries the PID sets for the current tick (proc, user, slurm, all).
Use `ctx->all_pids` / `ctx->n_all` to iterate over all tracked processes.

---

## Adding a new backend

**1.** Create `metrics/<name>/<name>.h`:
```c
#ifndef JUMPER_METRICS_<NAME>_H
#define JUMPER_METRICS_<NAME>_H
#include "monitor.h"
extern const CCollector <name>_collector;
#endif
```

**2.** Implement `metrics/<name>/<name>.c` — define all needed callbacks and expose:
```c
const CCollector <name>_collector = {
    .name          = "<name>",
    .setup         = <name>_setup,
    .snapshot      = <name>_snapshot,
    .emit_columns  = <name>_emit_columns,
    .emit_sample   = <name>_emit_sample,
    .post_tick     = NULL,
    .teardown      = NULL,
};
```

**3.** Add to `Makefile`:
```makefile
SRCS := \
    ...
    metrics/<name>/<name>.c
```

**4.** Register in `monitor.c`:
```c
#include "metrics/<name>/<name>.h"

static const CCollector * const g_registry[] = {
    &cpu_collector, &memory_collector, &io_collector, &gpu_collector,
    &<name>_collector,
};
```

**5.** Add to `config/collectors/c/collectors.yaml`:
```yaml
collectors:
  - cpu
  - memory
  - io
  - gpu
  - <name>
```

After these changes, rebuild with `make` (or trigger `ensure_native_c(force_build=True)`
from Python) — the new collector will be active on the next monitor start.
