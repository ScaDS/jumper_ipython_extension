# Benchmark Replay Modes

`%perfmonitor_ai_review --benchmark` measures every suggestion by running it. To
run a cell at all it first has to rebuild the state that cell depends on, which
means replaying the cells before it. **The replay mode decides how often that
prefix is paid for.**

| Mode | Prefix cost | Languages | Status |
|------|-------------|-----------|--------|
| `full` | once per measurement | all | Default. The reference every other mode falls back to. |
| `fork` | once per benchmark | Python | Available. |
| `dill` | once per benchmark | Python | Not built yet. |

A benchmark takes `(1 + suggestions) × runs` measurements, plus one more for
every repair round. Under `full` each of those replays the entire prefix from a
cold interpreter, so a notebook whose second cell loads a large dataset spends
almost all of its benchmark time re-loading it.

`full` is the reference rather than a guarantee of correctness. It replays cells
through the same hooks a live notebook uses, but a replay is not a notebook: it
runs in a temporary working directory, has no IPython, and a prefix cell that
depended on either will behave differently there under every mode.

## Choosing a mode

```yaml
ai:
  benchmark:
    replay:
      mode: fork
```

or per run:

```
%perfmonitor_ai_review --cells 7 --benchmark --replay-mode fork
```

**When `fork` pays off:** an expensive prefix, and a cell whose work is arrays -
numpy, BLAS, anything where the data is large but the number of Python objects
is small. That is the case it was measured on and the case it reproduces.

**When to stay on `full`:** when the memory numbers are what you are after (see
the caveat below), or when the cell works over large Python data structures - a
list of millions of ints, a dict of objects. Those are the workloads where a
fork distorts timing most, and where the warning described below will fire.

## How `fork` is arranged

```
notebook kernel  →  supervisor  →  zygote  →  measurement child
                    (sampler)      (prefix)   (the cell)
```

The prefix runs once, in the zygote, which then forks a child per measurement. A
child inherits the whole state - every object, import and JIT-compiled function -
and dies with its mutations, so measurements stay isolated from each other.

The monitor lives one process higher, and that placement is the reason the
arrangement has three processes rather than two. A monitor runs a sampler thread,
and **only the forking thread survives a `fork()`** - a lock held by a vanished
thread stays locked forever in the child. Sampling from the supervisor keeps the
forking process free of threads of our making, and costs nothing in coverage:
sampling walks the whole process tree, so the measurement child is picked up as a
grandchild exactly as a child would be.

Each link asks the kernel to kill it when its parent goes away, and every
measurement runs in a process group of its own. An interrupted or restarted
notebook therefore does not leave a sampler and a full copy of the prefix behind,
and a cell killed by a timeout takes whatever it started with it.

## What `fork` checks before it trusts itself

**The verdict is structural: does a forked child get back the cores its parent
was using.** After a `fork()` only the calling thread remains, so a BLAS or
OpenMP pool built during the prefix is gone; whether the library rebuilds it is
what matters, because a cell timed on one core instead of twelve looks entirely
plausible and is badly wrong. The probe counts the threads the child reaches
after its own parallel region. On this machine a parent on 12 threads forks a
child that starts on 1 and returns to 12; under `OPENBLAS_NUM_THREADS=1` it reads
1 against 1 and correctly sees nothing lost. Recovering less than half refuses
the mode.

An earlier version decided this by comparing timings across the fork, and
**refused a healthy machine in 6 runs out of 20**: two unreplicated measurements
of a few milliseconds cannot separate a real effect from a busy machine. Those
timings are still taken and written to the debug log, because they are the only
way an unknown slowdown would ever surface - but they no longer decide anything.

## What each measurement pays, and what was moved out of it

A forked child carries costs that belong to *being a forked child*, not to the
cell. They do not average out: every measurement is a fresh child, so dropping
the first run as a warm-up does not help. Two are paid before the clock starts:

- **Inherited pages.** The first read of a copy-on-write page costs more than a
  later one. Measured: a child summing an inherited 160MB array took 13.0ms
  against its parent's 6.8ms, and 7.1ms once the pages had been walked first -
  while the same child summed its *own* fresh array at 6.8ms. The penalty scales
  with how much of the prefix's data a cell touches, so it falls hardest on fast
  vectorized rewrites, the very thing a review is looking for. Each child walks
  its resident inherited pages before the timer starts.
- **The thread pool.** Rebuilding it is lazy, so without help it happens on the
  cell's first parallel operation, inside the window. Measured: a matmul taking
  105ms in the parent took 132ms as the first one in a child, and 115ms after a
  0.8ms warm-up. The warm-up covers BLAS only; other runtimes' pools are not
  reached.

### What cannot be moved out

Reading a Python object **writes** to the page it lives on, because the reference
count lives inside the object - and a write to an inherited page is a real copy,
not bookkeeping. Walking ahead cannot pre-pay that; copying is exactly what a
fork avoids. Measured: an inherited 20-million-element list summed in 737ms
against 107ms on a second pass. Private file-backed mappings (`numpy.memmap`,
h5py, pyarrow) carry a similar first-touch cost - 20.8ms against a parent's
11.9ms on a 200MB mapping - and walking *them* was measured not to help.

So instead of being paid, this is counted. Every measurement reports the page
faults taken during the cell, and when they plausibly account for a twentieth of
it or more, a warning names the figure:

```
[JUmPER]: a benchmark measurement spent at least 12ms of its 93ms on 23535 page
faults - the cost of touching memory inherited from the prefix rather than of the
cell itself. ... Re-run with --replay-mode full for a number that does not carry this.
```

**The figure is a floor.** Faults are priced at a *read* fault's cost, measured
by the page walk on the same machine, while most of what is counted are writes,
which cost several times more. Pricing them as writes was tried and produced
estimates larger than the measurements they described, so the floor is reported
and the threshold set low instead.

## How closely `fork` and `full` agree

On array work, closely. Interleaving the two modes over 12 measurements each of a
2000×2000 matmul on an inherited matrix: median ratio 1.04, and `full`'s own runs
spanned 80–115ms, so the difference is comparable to the machine's own noise.

On Python-object work they do not agree, and the warning above is what says so.
No claim is made here that the modes are interchangeable; they agree where they
have been measured to agree.

## Fallback

A mode that cannot serve the cell in front of it **never fails the benchmark** -
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
  other is not a speedup - and a variant inheriting a fork-derived time budget
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
Without that, a mode that rebuilt the wrong state passes unnoticed - every
suggestion is compared against a baseline that went through the same broken
rebuild, so the two agree with each other and the divergence check reports a
match. It would cost one extra prefix run per benchmark.
