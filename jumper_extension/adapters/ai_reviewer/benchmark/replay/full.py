"""Replay the whole prefix in a fresh process, once per measurement.

The behaviour the benchmark shipped with, and the fallback every other strategy
degrades to. Each run is its own interpreter: the prefix rebuilds the state the
cell needs, and nothing a variant mutates can leak into the next one. It is
correct for every language and every workload, and it is the reason a benchmark
costs what it costs - the prefix is paid for again on each of the
``(1 + variants) x runs`` measurements, plus once per repair.
"""
import logging
import os
import subprocess
import time

from jumper_extension.adapters.ai_reviewer.benchmark.models import FAILED, OK, TIMEOUT
from jumper_extension.adapters.ai_reviewer.benchmark.replay.base import (
    FULL,
    ReplayResult,
    ReplayStrategy,
    tail,
)
from jumper_extension.adapters.ai_reviewer.language import ReplayRequest

logger = logging.getLogger("extension")


class FullReplayStrategy(ReplayStrategy):
    """Rebuild the state by re-running every prefix cell, every time."""
    name = FULL

    def replay(self, code: str, tag: str, timeout: float | None) -> ReplayResult:
        context = self.context
        session_path = os.path.join(context.work_dir, f"{tag}_session.zip")
        fingerprint_path = os.path.join(context.work_dir, f"{tag}_fingerprint.json")
        # A timed run always fingerprints its outputs: verification rides along
        # with the replay (the two are one check level) and is cheap beside it.
        output_names = context.adapter.output_names(code)
        artifact = context.adapter.render_replay(
            ReplayRequest(
                prefix_cells=context.prefix_cells,
                target_code=code,
                interval=context.interval,
                output_names=output_names,
                session_path=session_path,
                fingerprint_path=fingerprint_path,
                output_path=os.path.join(context.work_dir, tag),
                work_dir=context.work_dir,
            )
        )

        for stale in (session_path, fingerprint_path):
            if os.path.exists(stale):
                os.remove(stale)

        started = time.perf_counter()
        try:
            completed = subprocess.run(
                artifact.command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=context.work_dir,
                env=self.child_env(),
            )
        except subprocess.TimeoutExpired:
            return ReplayResult(
                status=TIMEOUT,
                error=f"Exceeded the {timeout:.0f}s budget and was killed.",
            )
        wall = time.perf_counter() - started

        if completed.returncode != 0:
            return ReplayResult(status=FAILED, error=tail(completed.stderr))
        if not os.path.exists(session_path):
            return ReplayResult(
                status=FAILED,
                error=f"The run produced no session export.\n{tail(completed.stderr)}",
            )
        return ReplayResult(
            status=OK,
            session_path=session_path,
            fingerprint_path=fingerprint_path,
            wall_s=round(wall, 4),
        )
