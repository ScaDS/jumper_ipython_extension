---
title: Query Console
---

# AI Reviewer — prompt assembly console

This console makes the prompt-assembly model operable. Pick a request, a
strategy, and whether `--note` is present; the store, the query plan, and the
two returned messages update together.

<iframe
  src="../query-console-embed.html"
  title="Prompt assembly query console"
  style="width: 100%; height: 1150px; border: 1px solid var(--md-default-fg-color--lightest); border-radius: 3px;"
  loading="lazy">
</iframe>

[Open the console in its own tab](query-console-embed.html){: target="_blank" }

## What the panels show

| Panel | Content |
|---|---|
| Sources | Every row of the store: recorded facts under root knowledge, prompt text under Text, each with the flag this request gives it |
| Query | The declared items as a plan: what is selected, what the scope is, what the strategy overrode, and what is never queried |
| Returned context | The system message with its `given` bullets and rules, and the human message with one payload block per selected identifier |

Strategy effects are the real ones: `deep` opens `raw_perf`, `r_clean` closes
`packages` and swaps the style rule, `parallelization` adds a hint to analyze
and the graphics-offload rule to suggest, and `custom` turns the note on.
