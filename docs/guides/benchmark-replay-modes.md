# Benchmark Replay Modes

`%perfmonitor_ai_review --benchmark` measures every suggestion by running it. To
run a cell at all it first has to rebuild the state that cell depends on, which
means replaying the cells before it. **The replay mode decides how often that
prefix is paid for.**

| Mode | Prefix cost | Languages | Use it when |
|------|-------------|-----------|-------------|
| `full` | once per measurement | all | Default. Always correct. |
| `fork` | once per benchmark | Python | The prefix is slow and the cell is CPU-bound. |
| `dill` | once per benchmark | Python | *Not yet built.* |

A benchmark takes `(1 + suggestions) × runs` measurements, plus one more for
every repair round. Under `full` each of those replays the entire prefix from a
cold interpreter, so a notebook whose second cell loads a large dataset spends
almost all of its benchmark time re-loading it.

## Choosing a mode

Set it in the configuration:

```yaml
ai:
  benchmark:
    replay:
      mode: fork
      cross_check: true
```

or per run:

```
%perfmonitor_ai_review --cells 7 --benchmark --replay-mode fork
```

## Fallback

A mode that cannot serve the cell in front of it **never fails the benchmark** -
it degrades to `full` and logs why. That happens when:

- the cell is not Python (`fork` is Python-only, so R cells always use `full`);
- the platform is not POSIX;
- the prefix left an initialized CUDA context, which no forked child can use;
- the fidelity probe (below) refuses;
- the zygote dies mid-benchmark, in which case the measurement that lost it is
  retried under `full` rather than blamed on the suggestion.

## What `fork` checks before it trusts itself

A forked child does not automatically run at the speed of the process that
forked it, and both ways it can differ are invisible in the results - the values
come out identical while the timings are wrong. Before serving a single
measurement, `fork` measures both across a real fork and refuses if either
fails:

- **compute** - only the forking thread survives a `fork()`, so a BLAS/OpenMP
  pool built during the prefix may come back at one thread. A cell timed on one
  core instead of twelve looks plausible and is badly wrong.
- **memory** - the first read of an inherited copy-on-write page costs more than
  a later read. Measured on one machine, a child summing an inherited 160MB
  array took 13.0ms against its parent's 6.8ms, while the same child summed its
  *own* fresh array at 6.8ms. That penalty scales with how much of the prefix's
  data a cell touches, so it falls hardest on fast vectorized rewrites - the very
  thing a review is looking for. Each child walks its inherited pages before the
  timer starts to move that cost out of the measured window, and this arm
  confirms the walk took.

With both in place, `fork` reproduces `full`'s numbers. Measured on a 160MB
array, `full` reported a 145x speedup for a vectorized rewrite and `fork`
reported 127x; before the page walk was added, `fork` reported 81x against
`full`'s 170x.

## Caveat: memory metrics under `fork`

!!! warning "RSS is not comparable with a full replay's"

    Memory is sampled as summed RSS over the process tree. A forked child's RSS
    includes every copy-on-write page it inherited from a zygote that is still
    holding the whole prefix, and the two are counted twice over. Under `fork`,
    memory readings describe **what the cell inherited, not what it allocated**,
    and a cell allocating 10MB on top of a 20GB prefix reports something close to
    20GB.

    **Timings are unaffected** - this is the metric, not the measurement. Use
    `full` when the memory numbers are what you are after. A warning is logged
    whenever `fork` is active.

    Reporting per-cell memory correctly under `fork` needs PSS/USS accounting
    rather than RSS, which is tracked as separate work.

## `cross_check`

Before a fast mode is trusted, the baseline is measured once through it and once
through `full`, and the results are compared. Without this, a restore that
rebuilt the wrong state passes unnoticed: every suggestion is compared against a
baseline that went through the same broken restore, so the two agree with each
other and the divergence check reports a match. It costs one extra prefix run per
benchmark.
