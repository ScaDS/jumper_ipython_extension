---
title: Steering the Review
---

# Steering the Review

A review is steered with two arguments: `--strategy` picks a prepared angle,
`--note` says anything the prepared ones do not cover.

```python
%perfmonitor_ai_review --strategy parallelization
%perfmonitor_ai_review --strategy custom --note "keep it single-threaded, we run under SLURM"
```

## What a strategy changes

Each strategy pulls two levers at once:

- **Context sources** — what the model is given (cell code, timings, metric
  summaries, tags, hardware, installed packages, raw metric arrays).
- **Prompt rules** — individual instructions in the system prompts, switched on
  or off one by one.

The two stay in step: a source that is turned off is neither collected nor
mentioned in the prompt, so the model is never told about data it did not get.

## Available strategies

| `--strategy` | Use it when | What it turns on |
|---|---|---|
| `faster` *(default)* | You just want the cell to be quicker. | The standard context; the model picks the angle. |
| `parallelization` | A GPU sits idle, or cores do. | GPU-offload rules, a hint to look for under-used parallel hardware, and the installed-package list. |
| `deep` | The summaries hide the shape of the problem — bursts, stalls, a slow start. | The **raw per-timestep metric arrays** behind `%perfmonitor_plot`, plus packages. |
| `custom` | None of the above describes the goal. | Nothing on its own — **requires `--note`**. |
| `r_clean` | Reviewing R cells. | The R style rule instead of the Python/PEP 8 one; drops the package list. |

The default comes from `ai.context.strategy` in
[configuration](../configuration.md#aicontext-what-the-model-is-told).

!!! tip
    `deep` sends every sample of every metric for the reviewed range. It is the
    most expensive strategy by a wide margin — use it on one cell, not a range.

## `--note`

Free text, folded into the prompt. Its meaning depends on where it is used:

| Command | Effect of `--note` |
|---|---|
| `%perfmonitor_ai_review --note "..."` | Steers which suggestions are generated. |
| `%perfmonitor_ai_review --resume ID --select N --note "..."` | Rewrites that one suggestion before it is applied. |

!!! note
    `--note` replaced the earlier `--refine`; the old flag no longer exists.

## Adding your own strategy

Strategies are data, not code. They live in
`jumper_extension/adapters/ai_reviewer/strategy/strategies.yaml`:

```yaml
  - id: memory
    name: Cut peak memory
    description: Trade speed for a smaller footprint.
    effect:
      overrides:
        gpu_offload: false        # prompt rules
      context:
        raw_perf: true            # context sources
```

Both buckets are `id: enabled` maps and are merged into one — `overrides` names
prompt items, `context` names sources. Add the entry and `--strategy memory` is
accepted immediately; the choices `--strategy` offers are read from this file.

Set `require_note: true` to make the strategy refuse to run without `--note`,
as `custom` does.

### The ids you can toggle

| Bucket | Ids |
|---|---|
| `context` | `code`, `timing`, `perf`, `raw_perf`, `hardware`, `tags`, `packages` |
| `overrides` | Any item id in the prompt specs — e.g. `gpu_offload`, `parallel_analysis_hint`, `pep8_multiline`, `r_style`, `preserve_comments`, `preserve_style` |

Prompt items live in `prompts/<analyze|suggest|refine|fix>/spec.yaml`, each
with an `id`, an `enabled` default and either inline `text` or a shared
`fragment` from `prompts/fragments/`. Editing `enabled` there changes the
default for every strategy; a strategy's `overrides` change it for that
strategy only.

!!! note
    `packages` reports the versions of the libraries listed under
    `ai.context.known_packages`. Extending that list is how you tell the model
    what it is allowed to suggest.
