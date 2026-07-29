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

        logger.debug(f"[JUmPER]: benchmark fork probe: {ready.get('detail', '')}")
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
        return ReplayResult(
            status=OK,
            session_path=session_path,
            fingerprint_path=fingerprint_path,
            wall_s=wall,
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
