---
title: AI Review
---

# AI Review

`%perfmonitor_ai_review` hands a monitored cell to an LLM. It reads the cell's
source together with the performance data JUmPER already collected, names the
bottleneck, and proposes ranked rewrites you can apply with one command — or,
with [`--benchmark`](benchmark.md), measure before you trust them.

## Prerequisites

1. **Install the optional extras**

    ```bash
    pip install 'jumper-extension[ai]'      # source checkout: pip install -e '.[ai]'
    ```

2. **Export the API key** named by `ai.llm.api_key_env` (default `JUMPER_AI_API_KEY`):

    ```bash
    export JUMPER_AI_API_KEY=...
    ```

3. **Have monitoring running** — the review is built from measured data, not from the code alone.

!!! note
    The key is the only secret read from the environment. Endpoint, model and
    limits are configuration, not environment variables — see
    [Configuration](../configuration.md#ai-configaidefaultyaml). Without the
    extras the magic prints an install hint and changes nothing else.

## The loop

```python
%load_ext jumper_extension
%perfmonitor_fast_setup      # starts monitoring

# ... run the cell you want to improve ...

%perfmonitor_ai_review                                  # 1. review it
%perfmonitor_ai_review --resume 7e5a8568 --select 1     # 2. apply option 1
```

Step 1 displays a card with a run id and numbered options. Step 2 places the
chosen code into the **next cell**, ready to inspect and run. Nothing is
executed or overwritten on your behalf.

## Choosing what to review

| Argument | Effect |
|---|---|
| *(none)* | The last cell long enough to have been measured. |
| `--cells 7` | One cell, by index. |
| `--cells 2:8`, `--cells :5`, `--cells 3:` | A range of cells. |
| `--level LEVEL` | Scope the metrics are read at (`process`, `user`, `system`, …); defaults to `perfreports.level`. |

Over a range each cell's source is marked with `# --- cell N ---` and timings
are given per cell, so a suggestion can name the cell it rewrites. Every
suggestion still rewrites **exactly one** cell.

!!! tip
    `%show_cell_history` lists the indices `--cells` expects.

## Reading the card

| Part | What it is |
|---|---|
| **run id** | Short id identifying this review; every follow-up command needs it. |
| **Analysis** | The bottleneck the model found. Its reasoning, if the model emits any, sits behind a collapsed spoiler. |
| **Options** | Ranked, most impactful first: title, one-line rationale, and a unified diff against your code. Over a range the title also names the target cell. |
| **Verdict** | Present only after [`--benchmark`](benchmark.md) — what the option actually measured. |
| **Resume commands** | Ready-to-paste commands for applying each option. |

## Applying a suggestion

```python
%perfmonitor_ai_review --resume 7e5a8568 --select 2
%perfmonitor_ai_review --resume 7e5a8568 --select 2 --note "use multiprocessing instead of joblib"
```

`--select N` takes the 1-based option number. With `--note` the option is
rewritten to your instruction first, then placed into the next cell.

!!! warning "Run ids live in the kernel's memory"
    Pending reviews are held by the running kernel, not on disk. After a
    restart the id is gone and the review has to be run again.

## Other languages

Under the [wrapper kernel](../../wrapper-kernel/guides/wrap-kernel.md) the
language is recorded per cell, so a mixed notebook is handled cell by cell.

| Language | Review | Benchmark |
|---|---|---|
| Python | yes | yes, all replay modes |
| R (`%wrap_kernel ir`) | yes | yes — syntax check and timed run, provided `Rscript` is on `PATH`. `fork` is Python-only, so R replays fall back to `full` |
| Anything else | yes | no adapter: benchmark steps are skipped with a warning |

!!! tip
    For R, add `--strategy r_clean` so suggestions come back idiomatic R
    instead of PEP 8-shaped Python. See [Steering the review](strategies.md).

## Troubleshooting

| Message | Cause |
|---|---|
| `AI review is not available: optional dependencies ... are not installed` | The `[ai]` extras are missing; the message carries the exact install command. |
| `No active performance monitoring session` | Start monitoring before reviewing. |
| `strategy 'custom' requires a --note instruction` | `custom` is free-form; say what you want with `--note`. |
| `No pending AI review found for run_id '...'` | Wrong id, or the kernel was restarted. |
| `--benchmark measures one cell, but this review covers N` | Benchmarking needs a single-cell review — narrow `--cells`. |

## Next steps

- [Steering the review](strategies.md) — pick what the model optimizes for.
- [Measuring suggestions](benchmark.md) — turn proposals into measured verdicts.
- [Replay modes](replay-modes.md) — make a benchmark cheaper.
