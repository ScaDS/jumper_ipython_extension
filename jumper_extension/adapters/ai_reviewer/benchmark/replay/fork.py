"""Drive a zygote that holds the prefix state, one forked child per measurement.

The parent half of the fork mode: it starts the zygote, asks it to replay a cell,
and hands the resulting session export back to the runner unchanged, so what the
benchmark reads is what it reads for every other strategy. The zygote decides
whether forking is safe at all - a GPU context or a thread pool that does not
survive the fork - and says so in its ready message; everything it refuses ends
up as a fallback to the full replay rather than as numbers nobody should trust.

Two costs disappear here and one caveat appears. Gone: the prefix, paid once
instead of per measurement, and the interpreter start behind it. The caveat is
memory. Metrics are summed RSS over the process tree, and a forked child's RSS
counts every copy-on-write page it inherited from a zygote that is still holding
the prefix - so under this mode memory readings describe what the cell inherited,
not what it allocated. Timing is unaffected; the warning below says as much.
"""
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from jumper_extension.adapters.ai_reviewer.benchmark.models import FAILED, OK
from jumper_extension.adapters.ai_reviewer.benchmark.replay.base import (
    FORK,
    PrepareOutcome,
    ReplayResult,
    ReplayStrategy,
    tail,
)
from jumper_extension.adapters.ai_reviewer.benchmark.script import build_prefix_script

logger = logging.getLogger("extension")

_ZYGOTE_MODULE = "jumper_extension.adapters.ai_reviewer.benchmark.replay.zygote"

# How long a stopping zygote gets to leave on its own before it is killed.
_STOP_TIMEOUT_S = 10.0

_RSS_CAVEAT = (
    "[JUmPER]: benchmark replay mode 'fork' is active: the prefix is replayed "
    "once instead of once per measurement. Timings are unaffected, but memory "
    "metrics are not comparable with a full replay's - a forked child's RSS "
    "includes the copy-on-write pages it inherited from the prefix."
)


class ForkReplayStrategy(ReplayStrategy):
    """Rebuild the state once, then fork a copy of it per measurement."""
    name = FORK
    languages = frozenset({"python"})

    def __init__(self, context):
        super().__init__(context)
        self._process = None
        self._responses = None
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
            return PrepareOutcome(False, f"the zygote could not be started: {error}")

        ready = self._receive()
        if ready is None:
            return PrepareOutcome(
                False,
                f"the zygote exited before it was ready.\n{tail(self._log_text())}",
            )
        if not ready.get("ok"):
            return PrepareOutcome(False, ready.get("reason", "the zygote refused to serve"))

        logger.debug(f"[JUmPER]: benchmark fork probe: {ready.get('detail', '')}")
        logger.warning(_RSS_CAVEAT)
        return PrepareOutcome(True)

    def replay(self, code: str, tag: str, timeout: float | None) -> ReplayResult:
        broken = self._health()
        if broken:
            return ReplayResult(status=FAILED, error=broken, strategy_broken=True)

        work_dir = self.context.work_dir
        session_path = os.path.join(work_dir, f"{tag}_session.zip")
        fingerprint_path = os.path.join(work_dir, f"{tag}_fingerprint.json")
        code_path = os.path.join(work_dir, f"{tag}.py")
        Path(code_path).write_text(code, encoding="utf-8")
        for stale in (session_path, fingerprint_path):
            if os.path.exists(stale):
                os.remove(stale)

        response = self._exchange(
            {
                "cmd": "replay",
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
            }
        )
        if response is None:
            return ReplayResult(
                status=FAILED,
                strategy_broken=True,
                error=f"the zygote stopped responding.\n{tail(self._log_text())}",
            )

        wall = float(response.get("wall_s") or 0.0)
        if response.get("status") != OK:
            return ReplayResult(
                status=response.get("status", FAILED),
                error=response.get("error", ""),
                wall_s=wall,
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

    def close(self):
        process, self._process = self._process, None
        if process is not None:
            self._stop(process)
        for name in ("_responses", "_log"):
            stream = getattr(self, name)
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
                setattr(self, name, None)

    def _spawn(self, script: str):
        # Output goes straight to a file, never a pipe: nobody drains the zygote
        # while it works, and a full pipe would wedge it mid-benchmark. It also
        # keeps stdout free to be what it is - the extension logs a line there on
        # import, and prefix cells print - while answers travel on a pipe of
        # their own that no logger can reach.
        self._log_path = os.path.join(self.context.work_dir, "zygote.log")
        self._log = open(self._log_path, "w")
        read_fd, write_fd = os.pipe()
        try:
            self._process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    _ZYGOTE_MODULE,
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
                    "--response-fd",
                    str(write_fd),
                ],
                stdin=subprocess.PIPE,
                stdout=self._log,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.context.work_dir,
                env=self.child_env(),
                pass_fds=(write_fd,),
            )
        finally:
            # The parent's copy has to go, or a dead zygote would never read as
            # EOF here: the pipe stays open as long as any writer holds it.
            os.close(write_fd)
        self._responses = os.fdopen(read_fd, "r")

    def _health(self) -> str:
        """Why the zygote cannot serve a request, or '' when it can."""
        if self._process is None:
            return "the zygote was never started."
        if self._process.poll() is not None:
            return f"the zygote exited with status {self._process.returncode}.\n{tail(self._log_text())}"
        return ""

    def _exchange(self, request: dict) -> dict | None:
        """Send one request and read its answer; None once the zygote is gone."""
        try:
            self._process.stdin.write(json.dumps(request) + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            return None
        return self._receive()

    def _receive(self) -> dict | None:
        """One response line, or None on EOF - which means the zygote died.

        No timeout of its own: the zygote enforces the budget it was given and
        always answers, so the only silence here is a dead process, and that
        closes the pipe.
        """
        try:
            line = self._responses.readline()
        except (OSError, ValueError):
            return None
        if not line:
            return None
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    def _stop(self, process: subprocess.Popen):
        try:
            if process.poll() is None:
                process.stdin.write(json.dumps({"cmd": "stop"}) + "\n")
                process.stdin.flush()
                process.wait(timeout=_STOP_TIMEOUT_S)
        except Exception:
            pass
        finally:
            if process.poll() is None:
                process.kill()
                try:
                    process.wait(timeout=_STOP_TIMEOUT_S)
                except Exception:
                    pass
            for stream in (process.stdin,):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass

    def _log_text(self) -> str:
        """Whatever the zygote wrote before it stopped - its own diagnostics."""
        try:
            if self._log is not None:
                self._log.flush()
            with open(self._log_path) as handle:
                return handle.read()
        except OSError:
            return ""
