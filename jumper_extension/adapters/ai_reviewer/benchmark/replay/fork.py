"""Measure by forking a process that already ran the prefix, instead of replaying it.

The notebook-side end of the fork mode. It starts a supervisor, asks it to
measure a cell, and hands the resulting session export back to the runner
unchanged, so what the benchmark reads is what it reads for every other
strategy. Behind the supervisor sits a zygote holding the prefix state; neither
is visible from here beyond the answers they give.

Whether forking is safe at all is decided down there, where the state actually
is - a GPU context, or a thread pool that does not survive a fork - and anything
refused comes back as a fallback to the full replay rather than as numbers
nobody should trust.

Two costs disappear and one caveat appears. Gone: the prefix, paid once instead
of once per measurement, and the interpreter start behind it. The caveat is
memory. Metrics are summed RSS over the process tree, and both the zygote
holding the prefix and the child that inherited it are in that tree - so under
this mode memory readings describe what the cell inherited, counted twice, not
what it allocated. Timing is unaffected; the warning below says as much.
"""
import logging
import os
import sys
import time
from pathlib import Path

from jumper_extension.adapters.ai_reviewer.benchmark.models import FAILED, OK
from jumper_extension.adapters.ai_reviewer.benchmark.replay.base import (
    FORK,
    PrepareOutcome,
    ReplayResult,
    ReplayStrategy,
    tail,
)
from jumper_extension.adapters.ai_reviewer.benchmark.replay.link import JsonChannel
from jumper_extension.adapters.ai_reviewer.benchmark.script import build_prefix_script

logger = logging.getLogger("extension")

_SUPERVISOR_MODULE = "jumper_extension.adapters.ai_reviewer.benchmark.replay.supervisor"

# How long a stopping supervisor gets to leave on its own before it is killed.
_STOP_TIMEOUT_S = 10.0

# Room for the two hops below this one to do their own work - the supervisor's
# allowance for sampling and export, and the round trips between them. This only
# has to outlast a healthy measurement; the budget a cell is actually held to is
# enforced where the cell runs.
_RELAY_ALLOWANCE_S = 300.0

# A measurement whose page faults plausibly cost this much of its own duration is
# worth telling the user about: the distortion falls hardest on the fastest
# rewrites, which are the ones a review exists to find.
_FAULT_SHARE_WARNING = 0.1

_RSS_CAVEAT = (
    "[JUmPER]: benchmark replay mode 'fork' is active: the prefix is replayed "
    "once instead of once per measurement. Timings are unaffected, but memory "
    "metrics are not comparable with a full replay's - the process tree holds "
    "both the prefix and a child that inherited it, and their RSS is summed."
)


def _deadline(timeout: float | None) -> float | None:
    """When to stop waiting on the supervisor, or None to wait as long as it lives.

    A cell with no budget of its own gets none here either: how long a baseline
    may take is the user's call, and the full replay does not cap it. A dead
    supervisor is still noticed in that case - that is what the channel watches
    for, and it is the failure this exists to stop hanging on.
    """
    if not timeout:
        return None
    return time.monotonic() + float(timeout) + _RELAY_ALLOWANCE_S


class ForkReplayStrategy(ReplayStrategy):
    """Rebuild the state once, then measure a forked copy of it each time."""
    name = FORK
    languages = frozenset({"python"})

    def __init__(self, context):
        super().__init__(context)
        self._channel = None
        self._log = None
        self._log_path = ""
        # Seconds per page fault on this machine, measured by the probe. Zero
        # means it could not be measured, and no fault warning is worth stating
        # without it - a fault count on its own says nothing about a cell.
        self._page_cost_s = 0.0

    def prepare(self) -> PrepareOutcome:
        if os.name != "posix":
            return PrepareOutcome(False, "forking needs a POSIX platform")

        try:
            script = build_prefix_script(
                self.context.prefix_cells,
                os.path.join(self.context.work_dir, "zygote_prefix.py"),
            )
            self._spawn(script)
        except Exception as error:
            return PrepareOutcome(False, f"the supervisor could not be started: {error}")

        ready = self._channel.receive()
        if ready is None:
            return PrepareOutcome(
                False,
                f"the supervisor exited before it was ready.\n{tail(self._log_text())}",
            )
        if not ready.get("ok"):
            return PrepareOutcome(False, ready.get("reason", "the supervisor refused to serve"))

        self._page_cost_s = float(ready.get("page_cost_s") or 0.0)
        logger.debug(
            f"[JUmPER]: benchmark fork probe: threads {ready.get('threads', '?')}, "
            f"timings {ready.get('timings', {})}, "
            f"{ready.get('walk_pages', 0)} pages walked per measurement in "
            f"{ready.get('walk_s', 0)}s"
        )
        logger.warning(_RSS_CAVEAT)
        return PrepareOutcome(True)

    def replay(self, code: str, tag: str, timeout: float | None) -> ReplayResult:
        if self._channel is None or not self._channel.alive():
            return ReplayResult(
                status=FAILED,
                strategy_broken=True,
                error=f"the supervisor is not running.\n{tail(self._log_text())}",
            )

        work_dir = self.context.work_dir
        session_path = os.path.join(work_dir, f"{tag}_session.zip")
        fingerprint_path = os.path.join(work_dir, f"{tag}_fingerprint.json")
        code_path = os.path.join(work_dir, f"{tag}.py")
        Path(code_path).write_text(code, encoding="utf-8")
        for stale in (session_path, fingerprint_path):
            if os.path.exists(stale):
                os.remove(stale)

        response = self._channel.ask(
            {
                "cmd": "measure",
                "code_path": code_path,
                "session_path": session_path,
                "fingerprint_path": fingerprint_path,
                "markers_path": os.path.join(work_dir, f"{tag}.markers.json"),
                "error_path": os.path.join(work_dir, f"{tag}.error.txt"),
                "stdout_path": os.path.join(work_dir, f"{tag}.stdout.log"),
                # A timed run always fingerprints its outputs: verification rides
                # along with the replay and is cheap beside it.
                "output_names": self.context.adapter.output_names(code),
                "timeout": timeout,
            },
            # The supervisor already allows for its own overhead on top of the
            # cell's budget; this only has to outlast that, and never expire on a
            # measurement that is merely slow.
            deadline=_deadline(timeout),
        )
        if response is None:
            return ReplayResult(
                status=FAILED,
                strategy_broken=True,
                error=(
                    f"the supervisor gave no answer: {self._channel.failure()}.\n"
                    f"{tail(self._log_text())}"
                ),
            )
        return self._result_of(response, session_path, fingerprint_path)

    def close(self):
        if self._channel is not None:
            self._channel.close(_STOP_TIMEOUT_S)
            self._channel = None
        if self._log is not None:
            try:
                self._log.close()
            except OSError:
                pass
            self._log = None

    def _result_of(
        self,
        response: dict,
        session_path: str,
        fingerprint_path: str,
    ) -> ReplayResult:
        """Turn the supervisor's answer into the runner's own vocabulary."""
        wall = float(response.get("wall_s") or 0.0)
        if response.get("status") != OK:
            return ReplayResult(
                status=response.get("status", FAILED),
                error=response.get("error", ""),
                wall_s=wall,
                # The supervisor says so when the failure is the zygote's rather
                # than the cell's, and that must not be blamed on a suggestion.
                strategy_broken=bool(response.get("gone")),
            )
        if not os.path.exists(session_path):
            return ReplayResult(
                status=FAILED,
                error="The run produced no session export.",
                wall_s=wall,
            )
        self._warn_if_distorted(response)
        return ReplayResult(
            status=OK,
            session_path=session_path,
            fingerprint_path=fingerprint_path,
            wall_s=wall,
        )

    def _warn_if_distorted(self, response: dict):
        """Say so when a measurement was mostly the fork's cost, not the cell's.

        A forked child pays for the memory it inherited the first time it touches
        it, and for Python objects that is a real copy: merely *reading* a list
        writes a reference count into a shared page. Measured here, an inherited
        20-million-element list summed in 737ms against the 107ms it took on a
        second pass. Walking the pages beforehand cannot prevent it - a copy is
        exactly what walking avoids - so the honest thing is to say that this
        particular number is inflated, and by roughly how much.

        Stated as a share of the measurement rather than as a fault count,
        because 20,000 faults decide everything for a 17ms cell and are nothing
        for a ten-second one. The price of a fault is the one the probe measured
        on this machine, not a number chosen in advance.
        """
        faults = int(response.get("faults") or 0)
        duration = float(response.get("duration_s") or 0.0)
        if not faults or not duration or not self._page_cost_s:
            return
        cost = faults * self._page_cost_s
        share = cost / duration
        if share < _FAULT_SHARE_WARNING:
            return
        logger.warning(
            f"[JUmPER]: a benchmark measurement spent about {cost * 1000:.0f}ms of "
            f"its {duration * 1000:.0f}ms on {faults} page faults - the cost of "
            "touching memory inherited from the prefix rather than of the cell "
            "itself. Cells working over large Python objects are affected most, "
            "and the timing understates them. Re-run with --replay-mode full for "
            "a number that does not carry this."
        )

    def _spawn(self, script: str):
        # Output goes straight to a file, never a pipe: nobody drains the
        # supervisor while it works, and a full pipe would wedge it mid-benchmark.
        self._log_path = os.path.join(self.context.work_dir, "supervisor.log")
        self._log = open(self._log_path, "w")
        self._channel = JsonChannel(
            command=[
                sys.executable,
                "-m",
                _SUPERVISOR_MODULE,
                "--prefix-script",
                script,
                "--interval",
                str(self.context.interval),
                "--prefix-count",
                str(len(self.context.prefix_cells)),
                "--work-dir",
                self.context.work_dir,
                "--language",
                self.context.adapter.language,
            ],
            fd_flag="--response-fd",
            log=self._log,
            cwd=self.context.work_dir,
            env=self.child_env(),
        )

    def _log_text(self) -> str:
        """Whatever the supervisor wrote before it stopped - its own diagnostics."""
        try:
            if self._log is not None:
                self._log.flush()
            with open(self._log_path) as handle:
                return handle.read()
        except OSError:
            return ""
