"""The cells this experiment measures - and nothing about how they are measured.

Each case is a notebook fragment aimed at one specific property of a fast replay
mode: a prefix that builds state, and one or two target cells timed against it.
That is the shape `%perfmonitor_ai_review --benchmark` sees when a model hands it
a rewrite, which is why the cases are written as cells rather than as functions.

Three modes are compared. `full` re-runs the prefix per measurement and is the
reference. `fork` and `dill` both pay for the prefix once - and distort what they
measure in **opposite directions**, which is why they are worth putting on one
axis:

- `fork` inherits the prefix's process. Its costs are the ones that come with
  inheriting memory, and its gift is everything the prefix warmed up.
- `dill` inherits the prefix's *values* and nothing else. It has none of fork's
  memory costs, and none of the warm-up either: every measurement is a cold
  process, so a first call the prefix should have paid for lands inside the
  timed window.

The cases are grouped by which mode they are aimed at (`aims_at`), and the
`expectation` on each is what `agents/benchmark_fork_replay_worklog.md` and
`agents/benchmark_dill_replay_plan.md` predict - recorded here so the report can
say whether this machine reproduces it rather than merely showing numbers.
"""
from dataclasses import dataclass, field

# Substituted, not formatted: cell sources contain braces, and a `.format` over
# user code would break on the first dict literal.
FIXTURE_TOKEN = "$FIXTURE"


@dataclass
class Case:
    """One prefix, and the cells timed against it under every replay mode."""
    name: str
    # The property under test, one line, quoted directly in the report.
    probes: str
    # What the worklog says should happen, so the report can confirm or refute it.
    expectation: str
    prefix: list[str]
    # Which fast mode this case was written to stress: "fork", "dill", "both" or
    # "neither" for the control. The report groups by it; nothing else reads it.
    aims_at: str = "fork"
    # label -> cell source. More than one only where the *ratio* between two
    # cells is the thing under test, since that ratio is what a benchmark reports.
    targets: dict[str, str] = field(default_factory=dict)
    # Elements of float64 to lay down on disk before the run, for cases that
    # measure a file-backed mapping. The prefix reaches the file through
    # FIXTURE_TOKEN; building it inside the prefix would have the full replay
    # rewrite it once per measurement and time the disk instead of the cell.
    fixture_elements: int = 0
    # Per-measurement budget, when the global one is wrong for this case. Set it
    # where a mode is expected to *hang* rather than to be slow: the budget is
    # then how long the experiment waits to find that out, and every round waits
    # again.
    timeout_s: float = 0.0

    def prefix_cells(self, fixture_path: str = "") -> list[dict]:
        """The prefix in the shape `BenchmarkRunner` expects."""
        return [
            {
                "index": index,
                "raw_cell": _resolve(source, fixture_path),
                "cell_magics": [],
            }
            for index, source in enumerate(self.prefix)
        ]

    def resolved_targets(self, fixture_path: str = "") -> dict:
        return {
            label: _resolve(source, fixture_path)
            for label, source in self.targets.items()
        }


def _resolve(source: str, fixture_path: str) -> str:
    return source.replace(FIXTURE_TOKEN, fixture_path)


ARRAY_REDUCE = Case(
    name="array_reduce",
    aims_at="neither",
    probes=(
        "the case fork was built for: a large numpy array in the prefix, and a "
        "cell whose work is BLAS rather than Python objects"
    ),
    expectation=(
        "all three modes agree - worklog SS4.4 measured a median fork ratio of 1.04 "
        "on this exact shape, within the machine's own noise, and an array is the "
        "one thing a checkpoint carries faithfully. The one thing dill loses here "
        "is the BLAS pool the prefix built, which this target is long enough to "
        "hide"
    ),
    prefix=[
        "import numpy as np",
        "rng = np.random.default_rng(0)",
        # 32MB, and the matmul builds the BLAS thread pool in the parent, so a
        # forked child faces a real pool to lose rather than an unbuilt one.
        "matrix = rng.random((2000, 2000))\nwarm = matrix @ matrix",
    ],
    targets={"matmul": "product = matrix @ matrix"},
)

PYTHON_OBJECTS = Case(
    name="python_objects",
    probes=(
        "reading an inherited Python object writes to its page, because the "
        "reference count lives inside the object - a write to a copy-on-write "
        "page is a real copy, and walking the pages ahead cannot pre-pay it"
    ),
    expectation=(
        "fork is several times slower and the per-measurement page-fault warning "
        "fires - worklog SS4.3 measured 17.7ms against 85.0ms on a 3M-element "
        "list. dill has no inherited pages at all, so it should read the same as "
        "full - and pay for it in wall time instead, unpickling four million "
        "integer objects before every measurement"
    ),
    prefix=[
        # ~145MB: 4M distinct int objects above the small-int cache, plus the
        # list's own pointer array.
        "values = [index * 3 for index in range(4_000_000)]",
    ],
    targets={"sum": "total = sum(values)"},
)

MEMMAP_SCAN = Case(
    name="memmap_scan",
    probes=(
        "a private file-backed mapping is faulted in again per child, and "
        "populating those mappings in the page walk was measured to make it worse"
    ),
    expectation=(
        "fork is roughly 1.5-1.8x slower - worklog SS4.3 measured 11.9ms against "
        "20.8ms on a 200MB mapping - and the walk does not help. dill cannot serve "
        "this prefix at all: a memmap owns an `mmap.mmap`, which refuses to pickle, "
        "so the mode falls back before writing anything and its column here is "
        "full in disguise. That is the right answer - a mapping restored as an "
        "ordinary array would time a different question - but it arrives as a "
        "traceback rather than as the named binding the guard produces for an "
        "open file"
    ),
    prefix=[
        "import numpy as np",
        f'mapping = np.memmap("{FIXTURE_TOKEN}", dtype="float64", mode="r", '
        f"shape=(25_000_000,))",
        # Read it through once so the pages are resident in the parent: the
        # question is what a *child* pays for them, not what a cold cache costs.
        "warm = float(mapping.sum())",
    ],
    targets={"scan": "total = float(mapping.sum())"},
    fixture_elements=25_000_000,
)

FAST_REWRITE = Case(
    name="fast_rewrite",
    probes=(
        "a benchmark reports a ratio, not a time: fork's fixed cost is a rounding "
        "error against a slow baseline and the whole of a fast rewrite, so the "
        "speedup - the number the user acts on - is where the distortion lands"
    ),
    expectation=(
        "the modes report different speedups for the same rewrite. Before the "
        "page walk landed, worklog SS4.1 measured 170x falling to 81x; with the "
        "walk in place this case measures how much of that is left. dill's fixed "
        "cost sits outside the timed window, so its speedup should track full's - "
        "unless the sub-millisecond rewrite is dominated by a first call the cold "
        "process has to make"
    ),
    prefix=[
        "import numpy as np",
        # 16MB: small enough that the vectorized rewrite lands near a millisecond,
        # which is where a fixed per-child cost distorts a ratio most.
        "data = np.arange(2_000_000, dtype=np.float64)",
    ],
    targets={
        "loop": "total = 0.0\nfor value in data:\n    total += float(value)",
        "vectorized": "total = float(data.sum())",
    },
)

THREAD_POOL = Case(
    name="thread_pool",
    aims_at="fork",
    probes=(
        "only the forking thread survives a fork(), and the child's warm-up covers "
        "BLAS and only BLAS - a pool belonging to another runtime is rebuilt "
        "lazily, inside the measured window (documented in zygote.warm_thread_pool)"
    ),
    expectation=(
        "fork is slower by the cost of rebuilding torch's pool. Worklog SS4.2 "
        "measured the same shape on BLAS - 105ms against 132ms cold - before the "
        "warm-up was added; nothing warms this one. Agreement instead would mean "
        "torch shares the pool numpy's warm-up already rebuilds. What was actually "
        "measured is worse than either: see fork_safety_repro.py. This is the "
        "prefix the guide sends to dill instead - a fresh process has no inherited "
        "pool to deadlock on, and pays a cold torch import per measurement for it"
    ),
    prefix=[
        # CPU only, on purpose: gpu_blocker refuses the mode outright once a CUDA
        # context exists, and this case is about threads, not accelerators.
        "import torch",
        "torch.set_num_threads(6)",
        (
            "left = torch.rand(2000, 2000)\n"
            "right = torch.rand(2000, 2000)\n"
            "warm = left @ right"
        ),
    ],
    targets={"matmul": "product = left @ right"},
    # These cells run in ~0.3s, so anything near the budget is not a slow cell -
    # it is the measurement never coming back. Short, because every round pays it.
    timeout_s=15.0,
)

JIT_WARMUP = Case(
    name="jit_warmup",
    aims_at="dill",
    probes=(
        "a checkpoint carries values, not the process that gave them meaning: the "
        "compiled code behind a JIT-ed function is not a value, so the dispatcher "
        "comes back uncompiled and the first call rebuilds it - inside the window "
        "the benchmark is timing"
    ),
    expectation=(
        "the exact mirror of fork's understatement. Under full the prefix compiles "
        "the function again for every measurement, and under fork the child "
        "inherits the compiled code, so both time a warm call; dill times the "
        "compile. Measured directly on this machine before writing the case: the "
        "first call after a checkpoint round trip took 462ms and the second 1ms"
    ),
    prefix=[
        "import numpy as np",
        "from numba import njit",
        (
            "@njit\n"
            "def accumulate(values):\n"
            "    total = 0.0\n"
            "    for index in range(values.shape[0]):\n"
            "        total += values[index] * 1.000001\n"
            "    return total"
        ),
        "data = np.arange(2_000_000, dtype=np.float64)",
        # Compiling here is the whole point: the prefix leaves a warm function
        # behind, and the question is which modes still have it.
        "warm = accumulate(data)",
    ],
    targets={"call": "total = accumulate(data)"},
    # Compiling costs under a second; anything near this budget is a hang, and
    # numba's threading layer is not something a forked child is promised.
    timeout_s=30.0,
)

IMPORT_HEAVY = Case(
    name="import_heavy",
    aims_at="dill",
    probes=(
        "dill stores modules by reference, so a restore re-runs every import the "
        "prefix ran. A prefix whose cost *is* its imports is therefore traded for "
        "a restore of the same cost, and the mode buys nothing"
    ),
    expectation=(
        "dill's per-measurement wall time lands near full's rather than near "
        "fork's, and the strategy says so itself: it compares restore_s against "
        "prefix_s and warns once when the first is within 80% of the second. fork "
        "pays the imports once, in the zygote, and is the only mode that gains here"
    ),
    prefix=[
        # Chosen for import weight, not for what they compute: none of this is
        # touched by the cell under test.
        "import numpy as np",
        "import pandas as pd",
        "import scipy.stats",
        "import sklearn.linear_model",
        "import statsmodels.api as sm",
        # A few megabytes, so the checkpoint is imports and almost nothing else.
        "values = np.arange(1_000_000, dtype=np.float64)",
    ],
    targets={"sum": "total = float(values.sum())"},
)

FAT_STATE = Case(
    name="fat_state",
    aims_at="dill",
    probes=(
        "what dill saves is the prefix's compute, minus the cost of storing its "
        "data - so a prefix that is cheap to run and fat to store is the shape "
        "where the trade goes the wrong way"
    ),
    expectation=(
        "no gain, and possibly a loss: the plan measured a 0.70s prefix leaving "
        "320MB going from 3.77s to 3.24s per measurement while the whole run went "
        "11.2s -> 14.5s. Cell timings should still agree with full - this is a "
        "cost finding, not a fidelity one, and it is the one the per-measurement "
        "charts cannot show"
    ),
    prefix=[
        "import numpy as np",
        "rng = np.random.default_rng(0)",
        # 320MB in about a third of a second: the worst possible ratio for a mode
        # that pays by the byte and is repaid by the second.
        "data = rng.random(40_000_000)",
    ],
    targets={"sum": "total = float(data.sum())"},
)

VIEW_ALIAS = Case(
    name="view_alias",
    aims_at="dill",
    probes=(
        "a view is aliasing, and aliasing is a property of memory rather than a "
        "value: `window = base[:n]` comes back from a checkpoint as an "
        "independent array, so a cell that writes through one and reads the other "
        "computes something else"
    ),
    expectation=(
        "the only case here whose finding is a wrong *answer* rather than a wrong "
        "time. full and fork share the memory and see the write; dill does not, "
        "and its fingerprints should differ from full's - which is precisely what "
        "ai.benchmark.replay.cross_check exists to catch, and what nothing inside "
        "the mode can"
    ),
    prefix=[
        "import numpy as np",
        # ones rather than zeros: a zeroed allocation is lazy, so the target would
        # be paying first-touch faults under full and not under dill, and the
        # timing would carry a second story on top of the one this case is about.
        "base = np.ones(10_000_000, dtype=np.float64)",
        # A view over the first half. Under every mode but dill it stays one.
        "window = base[:5_000_000]",
    ],
    targets={"write_through": "base += 1.0\ntotal = float(window.sum())"},
)

OPEN_HANDLE = Case(
    name="open_handle",
    aims_at="dill",
    probes=(
        "an open file pickles without complaint and restores as something else - "
        "a `w+` handle is truncated on disk - so the refusal has to be made inside "
        "the pickler rather than by trusting a successful dump"
    ),
    expectation=(
        "dill refuses this prefix before writing anything and the benchmark "
        "degrades to full, naming the binding. That is the intended outcome, so "
        "the dill column here is full in disguise and the report flags it as "
        "degraded. fork inherits the descriptor and is unaffected"
    ),
    prefix=[
        "import numpy as np",
        "values = np.arange(2_000_000, dtype=np.float64)",
        # Relative to the replay's own working directory, which every mode gives
        # the prefix - nothing outside the experiment is touched.
        'note = open("prefix_note.txt", "w")',
        'note.write("left open on purpose\\n")',
    ],
    targets={"sum": "total = float(values.sum())"},
)

RNG_STREAM = Case(
    name="rng_stream",
    aims_at="dill",
    probes=(
        "global random state is process state, not a binding: a restored process "
        "reseeds from OS entropy, and a benchmark that cannot reproduce its own "
        "random numbers reports correct suggestions as wrong"
    ),
    expectation=(
        "the trap dill handles rather than one it falls into - the checkpoint "
        "captures the RNG explicitly and restores it last, immediately before the "
        "cell. All three modes should fingerprint identically. A mismatch here "
        "would be a defect, not a distortion"
    ),
    prefix=[
        "import numpy as np",
        "import random",
        "np.random.seed(20260806)",
        "random.seed(20260806)",
        # Consume part of the stream, so what matters is the state mid-stream and
        # not merely that the same seed was set.
        "burned = np.random.random(1000).sum()",
    ],
    # Two bindings, not one sum of both. The fingerprint comparison is per name and
    # relative to 1e-6, so a stdlib draw of at most 1.0 folded into a numpy sum of
    # about a million would be a divergence three orders of magnitude below the
    # tolerance - present in the data and invisible to the check that matters.
    targets={
        "draw": (
            "numpy_draw = float(np.random.random(2_000_000).sum())\n"
            "python_draw = random.random()"
        )
    },
)

SLOW_PREFIX = Case(
    name="slow_prefix",
    aims_at="dill_gain",
    probes=(
        "the shape dill was built for and the rest of this file does not contain: "
        "a prefix that is expensive to *run* and cheap to *store*, so replacing a "
        "replay with a restore replaces seconds with milliseconds"
    ),
    expectation=(
        "the case measured in the dill plan - a 6s eigendecomposition leaving 11MB "
        "took a benchmark from 8.78s to 3.01s per measurement and 27.2s to 18.3s "
        "overall. Break-even should arrive on the second measurement, so dill "
        "should be cheaper than full here from the first round onwards"
    ),
    prefix=[
        "import numpy as np",
        "rng = np.random.default_rng(0)",
        "matrix = rng.random((1200, 1200))",
        "symmetric = matrix @ matrix.T",
        # ~4s of LAPACK on this machine, and the result is 11MB: the whole point.
        "values, vectors = np.linalg.eigh(symmetric)",
    ],
    # Deliberately not a matmul. A BLAS target in a restored process pays for the
    # thread pool the prefix built and dill did not inherit, which is a real effect
    # - it is measured in `jit_warmup` and again in `model_fit` - but it would put
    # it on top of the one case meant to show the cost win on its own.
    targets={"norms": "column_norms = np.sqrt((vectors ** 2).sum(axis=0))"},
)

MODEL_FIT = Case(
    name="model_fit",
    aims_at="dill_gain",
    probes=(
        "the same trade with an honest restore cost: a fitted model is small, but "
        "putting it back means importing sklearn again in every measurement, "
        "because dill stores modules by reference"
    ),
    expectation=(
        "dill wins anyway. The restore pays roughly a second of imports where the "
        "prefix paid several seconds of fitting, so the gain is the difference "
        "rather than the whole prefix - the general case, and the one the "
        "break-even arithmetic in the report is written for"
    ),
    prefix=[
        "import numpy as np",
        "from sklearn.ensemble import HistGradientBoostingClassifier",
        "rng = np.random.default_rng(1)",
        "features = rng.random((200_000, 20))",
        "labels = (features[:, 0] + features[:, 1] > 1).astype(int)",
        # A few seconds of fitting for a 1.4MB model.
        "model = HistGradientBoostingClassifier(max_iter=600).fit(features, labels)",
        "sample = features[:50_000]",
    ],
    targets={"predict": "predictions = model.predict(sample)"},
)

SIMULATION = Case(
    name="simulation",
    aims_at="dill_gain",
    probes=(
        "the extreme of the same trade: seconds of iteration leaving a few "
        "megabytes and importing nothing but numpy, which is the largest ratio of "
        "prefix cost to restore cost this file can construct"
    ),
    expectation=(
        "the clearest win available. Nothing here has to be re-imported and there "
        "is almost nothing to unpickle, so the restore should cost single-digit "
        "milliseconds against several seconds of replay, and the one-off setup "
        "should be repaid by the first measurement"
    ),
    prefix=[
        "import numpy as np",
        "rng = np.random.default_rng(2)",
        "state = np.zeros(500_000)",
        # Iterative on purpose: the work cannot be vectorized away, so the prefix
        # is genuinely slow while its result stays 4MB.
        (
            "for step in range(1100):\n"
            "    state = state * 0.999 + rng.random(500_000) * 0.001"
        ),
    ],
    targets={"sort": "total = float(np.sort(state).sum())"},
)

CASES: dict[str, Case] = {
    case.name: case
    for case in (
        ARRAY_REDUCE,
        PYTHON_OBJECTS,
        MEMMAP_SCAN,
        FAST_REWRITE,
        THREAD_POOL,
        JIT_WARMUP,
        IMPORT_HEAVY,
        FAT_STATE,
        VIEW_ALIAS,
        OPEN_HANDLE,
        RNG_STREAM,
        SLOW_PREFIX,
        MODEL_FIT,
        SIMULATION,
    )
}
