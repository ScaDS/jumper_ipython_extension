#!/usr/bin/env python3
"""What a forked child does not inherit, pinned down with no JUmPER involved.

Separate from the benchmark on purpose: both findings below turned up in the
experiment, and a finding that can be blamed on the harness is not a finding. Run
them all:

    python fork_safety_repro.py all

    numpy pool used by the parent, child does numpy    -> works
    torch imported but never used, child does torch    -> works
    torch imported AND USED by the parent, child torch -> HANGS
    torch used by the parent, child does numpy only    -> works
    parent seeds both RNGs, children draw              -> stdlib random RESEEDS

The first four are the `thread_pool` case. It was written to ask whether `fork`
*mis-times* a cell whose parallelism is not BLAS - the warm-up in
`zygote.warm_thread_pool` covers BLAS and says so. What it found is worse and
simpler.

The last line is the one that matters for the replay mode. `probe_fork` decides
whether forking is safe by forking a child that does **numpy** work; on this
machine that child comes back happily from a parent full of torch threads. So the
probe admits the mode, and the deadlock arrives later, on the first measurement
that touches torch.

Two consequences for the benchmark, both worth fixing:

- the admission probe cannot see this class of failure, because it exercises only
  the runtime it happens to import itself;
- the baseline is measured with no timeout at all (`orchestrator._run_timed` passes
  `timeout=None`, deliberately: how long a baseline may take is the user's call),
  and the channel's liveness poll cannot help - the zygote and its child are alive,
  merely blocked in a futex forever. A notebook whose prefix used torch would hang
  in `--benchmark --replay-mode fork` with nothing printed.

This is not a bug in JUmPER's fork mode as such: OpenMP runtimes are documented as
unsafe to use after `fork()`, and torch brings one. It is a bug in trusting a probe
that cannot ask the question.

The fifth stage is the `rng_stream` case, and it is CPython's own doing:
`random.py` ends with `_os.register_at_fork(after_in_child=_inst.seed)`, so the
module-level generator is **reseeded from OS entropy in every forked child**. numpy's
global stream has no such hook and survives. For the replay mode that means a cell
drawing from `random` gets a different number in every measurement under `fork`,
while `full` replays the seed and `dill` restores the captured state - so the
benchmark's own divergence check would report correct suggestions as wrong. Nothing
warns, and no probe looks.
"""
import os
import subprocess
import sys
import time

HANG_BUDGET_S = 15.0


def probe(label: str, work) -> None:
    """Fork, run *work* in the child, and report whether the child came back."""
    threads = len(os.listdir("/proc/self/task"))
    pid = os.fork()
    if pid == 0:
        try:
            started = time.time()
            work()
            # flush explicitly: os._exit skips the teardown that would do it, and
            # a forked child must not run the parent's atexit handlers.
            print(f"  {label}: child finished in {time.time() - started:.3f}s", flush=True)
            os._exit(0)
        except BaseException as failure:
            print(f"  {label}: child raised {type(failure).__name__}: {failure}", flush=True)
            os._exit(1)

    deadline = time.time() + HANG_BUDGET_S
    while True:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            print(f"  parent had {threads} threads; child exited status={status}")
            return
        if time.time() > deadline:
            print(
                f"  HUNG after {HANG_BUDGET_S:.0f}s: parent had {threads} threads, "
                f"child has {len(os.listdir(f'/proc/{pid}/task'))}, "
                f"blocked in {sorted(_wchans(pid))}"
            )
            os.kill(pid, 9)
            os.waitpid(pid, 0)
            return
        time.sleep(0.2)


def _wchans(pid: int) -> set:
    """What each of the child's threads is waiting on, per the kernel."""
    states = set()
    for task in os.listdir(f"/proc/{pid}/task"):
        try:
            with open(f"/proc/{pid}/task/{task}/wchan") as handle:
                states.add(handle.read().strip())
        except OSError:
            pass
    return states


def numpy_only():
    import numpy as np

    matrix = np.random.default_rng(0).random((1500, 1500))
    matrix @ matrix  # the parent builds its BLAS pool, as a prefix cell would
    print("numpy used by the parent, child does numpy:")
    probe("numpy in child", lambda: matrix @ matrix)


def torch_imported_only():
    import torch

    left, right = torch.rand(1500, 1500), torch.rand(1500, 1500)
    print("torch imported, parent never ran a parallel op, child does torch:")
    probe("torch in child", lambda: left @ right)


def torch_used():
    import torch

    left, right = torch.rand(1500, 1500), torch.rand(1500, 1500)
    left @ right  # this is what makes the difference
    print("torch imported AND used by the parent, child does torch:")
    probe("torch in child", lambda: left @ right)


def torch_used_numpy_child():
    import numpy as np
    import torch

    left, right = torch.rand(1500, 1500), torch.rand(1500, 1500)
    left @ right
    matrix = np.random.default_rng(0).random((1500, 1500))
    print("torch used by the parent, child does numpy only - what probe_fork does:")
    probe("numpy in child", lambda: matrix @ matrix)


def random_reseed():
    """Two children, one seeded parent: do they draw the same numbers?

    Deliberately not using `probe`, which reports only whether a child came back.
    What matters here is the value the child computed, so each child prints its own
    draws and the parent prints the numbers it would have drawn itself.
    """
    import random

    import numpy as np

    random.seed(20260806)
    np.random.seed(20260806)
    random.random(), np.random.random()  # burn one of each, as a prefix cell would

    # Flushed before forking, or each child inherits the buffered line and prints
    # the header again on its way out.
    print("parent seeded both generators; two children draw from each:", flush=True)
    for index in (1, 2):
        pid = os.fork()
        if pid == 0:
            print(
                f"  child {index}: random.random()={random.random():.12f}  "
                f"np.random.random()={np.random.random():.12f}",
                flush=True,
            )
            os._exit(0)
        os.waitpid(pid, 0)
    print(
        f"  parent : random.random()={random.random():.12f}  "
        f"np.random.random()={np.random.random():.12f}"
    )
    print(
        "  -> the numpy draws agree, the stdlib draws do not: random.py registers "
        "an at-fork handler that reseeds it."
    )


STAGES = {
    "numpy": numpy_only,
    "torch-import-only": torch_imported_only,
    "torch-used": torch_used,
    "torch-used-numpy-child": torch_used_numpy_child,
    "random-reseed": random_reseed,
}


def main(argv: list[str]) -> int:
    wanted = argv[1:] or ["all"]
    if wanted == ["all"]:
        # Each stage in its own interpreter: the point is what a *fresh* process
        # that imported one thing does, and imports cannot be undone.
        for name in STAGES:
            subprocess.run([sys.executable, __file__, name], check=False)
        return 0
    for name in wanted:
        if name not in STAGES:
            print(f"unknown stage {name}; known: {', '.join(STAGES)}, all")
            return 2
        STAGES[name]()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
