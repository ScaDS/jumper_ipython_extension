---
title: Replay Modes
---

# Replay Modes

To time a cell at all, [a benchmark](benchmark.md) first has to rebuild the
state that cell depends on, which means replaying the cells before it. **The
replay mode decides how often that prefix is paid for.**

| Mode | Prefix cost | Languages | Status |
|------|-------------|-----------|--------|
| `full` | once per measurement | all | Default. The reference every other mode falls back to. |
| `fork` | once per benchmark | Python | Available. |
| `dill` | once per benchmark | Python | Not built yet. |

Before repairs, a benchmark takes `(1 + suggestions) × runs` measurements. A
repaired suggestion may add up to `runs` more measurements per repair round.
Under `full` each measurement replays the entire prefix from a cold interpreter,
so a notebook whose second cell loads a large dataset spends almost all of its
benchmark time re-loading it.

## Choosing a mode

```yaml
benchmark:
  replay:
    mode: fork
```

or per run:

```python
%perfmonitor_ai_review --cells 7 --benchmark --replay-mode fork
```

| Choose | When |
|---|---|
| `fork` | An expensive prefix, and a cell whose work is **arrays** — numpy, BLAS, anything where the data is large but the number of Python objects is small. |
| `full` | The **memory numbers** are what you are after (see the caveat below), or the cell works over large **Python data structures** — a list of millions of ints, a dict of objects. |

!!! note
    `full` is the reference rather than a guarantee of correctness. It replays
    cells through the same hooks a live notebook uses, but a replay is not a
    notebook: it runs in a temporary working directory and has no IPython, so a
    prefix cell that depended on either behaves differently there under **every**
    mode.

## How `fork` is arranged

```
notebook kernel  →  supervisor  →  zygote  →  measurement child
                    (sampler)      (prefix)   (the cell)
```

The prefix runs once, in the zygote, which forks a child per measurement. A
child inherits the whole state — every object, import and JIT-compiled function
— and dies with its mutations, so measurements stay isolated from each other.

The monitor lives one process higher because **only the forking thread survives
a `fork()`**: a sampler thread in the forking process would leave the child
holding a lock nobody will ever release. Sampling from the supervisor costs
nothing in coverage — it walks the whole process tree.

Each link asks the kernel to kill it when its parent goes away, and every
measurement runs in its own process group, so an interrupted notebook does not
leave a sampler and a full copy of the prefix behind.

## What `fork` checks before it trusts itself

**The verdict is structural: does a forked child get back the cores its parent
was using.** After a `fork()` a BLAS or OpenMP pool built during the prefix is
gone, and a cell timed on one core instead of twelve looks entirely plausible
while being badly wrong. A probe counts the threads a child reaches after its
own parallel region; recovering less than half refuses the mode.

!!! note
    An earlier version decided this by comparing timings across the fork, and
    refused healthy machines regularly — two unreplicated measurements of a few
    milliseconds cannot separate a real effect from a busy machine. Those
    timings are still written to the debug log, but they no longer decide
    anything.

## What each measurement pays

A forked child carries costs that belong to *being a forked child*, not to the
cell, and they do not average out: every measurement is a fresh child. Two are
paid before the clock starts — each child walks its resident inherited pages,
and BLAS is warmed up so its thread pool is not rebuilt inside the timed window.
Other runtimes' pools are not reached.

One cost cannot be moved out. Reading a Python object **writes** to the page it
lives on, because the reference count lives inside the object — and a write to
an inherited page is a real copy. Walking ahead cannot pre-pay that; copying is
exactly what a fork avoids. Private file-backed mappings (`numpy.memmap`, h5py,
pyarrow) carry a similar first-touch cost.

So instead of being paid, it is counted. Every measurement reports the page
faults taken during the cell, and when they plausibly account for a twentieth
of it or more, a warning names the figure:

```
[JUmPER]: a benchmark measurement spent at least 12ms of its 93ms on 23535 page
faults - the cost of touching memory inherited from the prefix rather than of the
cell itself. ... Re-run with --replay-mode full for a number that does not carry this.
```

!!! warning "The figure is a floor, and the warning is not exhaustive"
    Faults are priced at a read fault's cost while most of what is counted are
    writes, which cost several times more. And a speedup can be understated
    without either measurement being distorted past the threshold — on
    Python-object work, expect `fork` to report a smaller gain than `full`
    would, with nothing firing to say so. On array work the two modes have been
    measured to agree within the machine's own noise.

## Fallback

A mode that cannot serve the cell in front of it **never fails the benchmark** —
it degrades to `full` and logs why. That happens when:

- the cell is not Python (`fork` is Python-only, so R cells always use `full`);
- the platform is not POSIX;
- the prefix left an initialized CUDA context, which no forked child can use.
  This check is best-effort: it asks torch whether its context is initialized,
  but refuses on a bare `import cupy` or `import jax` even for CPU-only work, and
  does not know about TensorFlow, numba.cuda, PyCUDA, ROCm, OpenCL, or `mpi4py`
  with MPI initialized;
- the thread probe refuses;
- **the supervisor or zygote dies mid-benchmark.** In that case everything
  measured so far is discarded and the whole benchmark re-runs on `full`. A
  benchmark reports ratios, so a variant timed one way over a baseline timed the
  other is not a speedup — and a variant inheriting a fork-derived time budget
  would be killed replaying a prefix that budget never allowed for.

## Caveat: memory metrics under `fork`

!!! warning "RSS is not comparable with a full replay's"

    Memory is sampled as summed RSS over the process tree. Under `fork` that tree
    holds both the zygote, which is keeping the whole prefix resident, and a child
    that inherited it - so the same pages are counted twice. A cell allocating
    10MB on top of a 20GB prefix reports something near 40GB.

    **Timings are unaffected** - this is the metric, not the measurement. Use
    `full` when the memory numbers are what you are after. A warning is logged
    whenever `fork` is active.

    Reporting per-cell memory correctly under `fork` needs PSS/USS accounting
    rather than RSS, which is tracked as separate work.

## `cross_check` — configured, not yet implemented

!!! danger "This setting does nothing today"

    `ai.benchmark.replay.cross_check` is accepted by the configuration and
    defaults to `true`, but **nothing reads it**. It is reserved for work that has
    not landed. Do not rely on it: no cross-checking of any kind currently
    happens, whatever it is set to.

What it is meant to become: before a fast mode is trusted, the baseline would be
measured once through it and once through `full`, and the results compared.
Without that, a mode that rebuilt the wrong state passes unnoticed — every
suggestion is compared against a baseline that went through the same broken
rebuild, so the two agree with each other and the divergence check reports a
match. It would cost one extra prefix run per benchmark.

!!! tip "Where the measurements behind this page live"
    The numbers each claim rests on — page-fault costs, thread-probe refusal
    rates, how closely `fork` and `full` agreed on which workloads — are recorded
    in the docstrings under
    `jumper_extension/adapters/ai_reviewer/benchmark/replay/`.
