# `full` against `fork` against `dill`: a concept run, not a test suite

Answers one question: **how comparable are the benchmark's three replay modes as
measuring instruments?** Fourteen cases — one control, four aimed at `fork`'s known
weak points, six at `dill`'s, and three where `dill` should pay for itself —
measured N times under each mode, plotted per case and summarised in one table.

Not pytest. Nothing here asserts; it measures and reports.

The two fast modes are cheap in the same way and wrong in opposite ways, which is
the reason to put them on one axis: `fork` inherits the whole process, so it pays
copy-on-write faults for what it reads and keeps every warm-up for free; `dill`
inherits the values and nothing else, so it pays for storing them and keeps no
warm-up at all.

Worth stating once, because it is easy to get backwards: an honest RSS, a
reproducible RNG and a torch-safe prefix are what `dill` has over **`fork`** —
`full` has all three too. Against `full`, `dill`'s only argument is wall time.

## Running it

```bash
# from the repository workspace root
venv/bin/python experiments/benchmark_modes/run_experiment.py --runs 11
jupyter lab experiments/benchmark_modes/report.ipynb   # run top to bottom
```

Or skip the first command: section 0 of the notebook runs the experiment itself
(`MEASURE = True`) and streams its output into the cell.

The figures are plotly — hover for exact times, click the legend to isolate a mode,
drag to zoom. Each is also written to `results/<run>/figures/*.png`; that export goes
through kaleido, which drives a headless Chrome. If PNGs are skipped with a message
about Chrome, run `python -c "import kaleido; kaleido.get_chrome_sync()"` once.

All fourteen cases at eleven rounds take about 30 minutes, of which nearly three are
`thread_pool` waiting out `fork`'s timeouts.

**Eleven runs is enough to see a 2x divergence, not a 20% one.** On this machine the
control case's fork/full ratio came out 0.92, 0.96 and 1.28 on three consecutive
five-run experiments, and 0.86 / 0.81 at eleven. Even at eleven the noise band is
around ±36%, so every figure draws it rather than leaving it implied. Differences
below that band are not findings, however consistent they look.

Useful flags:

| Flag | Why |
|---|---|
| `--runs N` | measurements per mode per target (default 5) |
| `--cases a,b` | subset of the fourteen; `python run_experiment.py --cases nope` lists them |
| `--timeout S` | per-measurement budget; a case may override it (`Case.timeout_s`) |
| `--prepare-timeout S` | how long a mode may take to set itself up before it is recorded unusable |
| `--keep-work` | keep every session export and replay script for debugging |
| `--out DIR` | write somewhere other than `results/<timestamp>/` |
| `--profile` | monitor the experiment with JUmPER itself — one virtual cell per case/mode/phase, written to `results/<run>/harness_session.zip` |
| `--profile-interval S` | sampling interval for `--profile` (default 0.25) |

The notebook reads the newest `results/*` directory unless `BENCHMARK_MODES_RESULTS`
names another, so an older run can be re-reported at any time.

## What lives where

| File | Responsibility |
|---|---|
| `cases.py` | the cells, which mode each is aimed at, and what each is expected to expose. No measuring. |
| `run_experiment.py` | the runs: interleaving, degradation detection, `runs.csv` + `meta.json`. No plotting. |
| `report.ipynb` | the report: interactive figures, tables, verdict. Can launch a run from its first cell. |
| `fork_safety_repro.py` | four forks, no JUmPER: why `thread_pool` produces timeouts rather than timings under `fork`. |
| `results/fixtures/` | on-disk data cases reuse between runs (a 191MB memmap file, not committed) |

## How it measures

- `BenchmarkRunner` is driven directly - no model, no orchestrator, no repair loop -
  so what is compared is the instrument, not the pipeline around it.
- **The modes are interleaved**, the order rotated each round, all three runners
  alive at once. A ratio between three blocks of measurements also contains whatever
  the machine's load did between the blocks.
- **Every row records the strategy that actually ran.** The replay registry degrades
  to `full` with a warning rather than failing, so without that column this
  experiment could show three series of which two are secretly the same one. Two
  cases (`open_handle`, `memmap_scan`) trigger that on purpose, because refusing is
  the right answer there.
- Four times are recorded per measurement (see "Reading the results"), because "is
  this the same instrument" and "what does a user wait through" are different
  questions with different answers.
- Three answers exist only in the extension's log or in a private file (the
  per-measurement page-fault estimate, the fork probe's verdict, the checkpoint's
  size), so a handler on the `extension` logger collects them per measurement rather
  than reaching into the strategy.

## Reading the results

`runs.csv` is one row per measurement:

| Column | What it is |
|---|---|
| `duration_s` | the target cell alone — the comparability question |
| `prefix_s` | what re-running the prefix cost this measurement: real under `full`, zero under the fast modes |
| `restore_s` | what loading the checkpoint cost this measurement: `dill` only |
| `harness_wall_s` | the whole measurement, interpreter start and session export included |

The notebook adds `rebuild_s = prefix_s + restore_s` — one column for "what this
measurement paid to have its state back", since the three modes pay it in three
different places. What the fast modes pay **once** instead is `prepare_s` in
`meta.json`.

`status` is the next column to look at: `ok`, `timeout` (the cell was killed),
`prepare_failed` (the mode never got as far as a measurement), `strategy_changed`
(it gave out mid-run). `degraded` is true whenever the strategy that ran is not the
mode that was asked for.

## Reading the JUmPER plots (`--profile`)

A cell on these plots is not a notebook cell. It is a *virtual block* — a labelled
span of time the run marked with `service.monitored(raw_cell=...)` so JUmPER records
it as if it were a cell. Two kinds exist:

- `prepare` — the mode's one-time setup, before anything is measured.
- `target=<label> run=<n>` — one whole measurement: state rebuild, the timed cell,
  interpreter start, session export.

**Every (case, mode) pair gives 12 blocks: 1 `prepare` + 11 `target`, one per round.**
`fast_rewrite` is the exception: two targets, so 1 + 22 = 23. Total 537.

The report draws one mode at a time and renumbers its blocks 0…N, so **index ranges
are identical for all three modes**; only the contents differ (medians, run
`20260813_133547`):

| Plot cell | Phase | `full` | `fork` | `dill` |
|---|---|---|---|---|
| `0` | `prepare` | 0.001 s — nothing to set up | 4.03 s — zygote + prefix | 3.30 s — prefix + checkpoint |
| `1`…`11` | `target`, one per round | 2.87 s | 0.18 s | 2.82 s |
| `1`…`22` | `fast_rewrite` only | 2.87 s | 0.18 s | 2.82 s |

`full`'s cell `0` is a sliver with no samples: it pays nothing once and everything per
measurement. `fork`'s cell `0` is its widest block — the prefix it never runs again.

In the session file the indices are interleaved instead (a mode's blocks run 178, 181,
183, 188 … across 0…536), because the modes take turns in rotating order. The last
plot in the notebook shows that raw order.

Two caveats:

- **Read level `user`, not `process`.** The experiment script itself only waits; the
  replays run in child processes. Under `fork`, the zygote (the parent holding the
  prefix) and its child are both alive, so their memory is counted twice.
- **Short blocks are nearly empty.** The sampler ticks every 0.25 s, so a 0.18 s
  `fork` measurement catches zero or one sample — all 12 `python_objects`/`fork`
  blocks hold 26 together. Only long blocks show shape: any `prepare` of a fast mode,
  any `target` under `full` or `dill`, and `thread_pool` under `fork` (15 s timeouts).
