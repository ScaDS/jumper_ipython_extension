"""Own the monitor, and keep it out of the process that forks.

The middle of the fork mode's three processes. Above it the strategy asks for a
cell to be measured; below it a zygote holds the prefix state and forks a child
to run that cell. This process does the one thing neither of them can:

    strategy  ->  supervisor  ->  zygote  ->  measurement child
                  (sampler)      (state)     (the cell)

A monitor runs a sampler thread, and only the forking thread survives a
``fork()`` - so a monitor must never live in the process that forks. Keeping it
here leaves the zygote with no threads of our making, and costs nothing in
coverage: sampling walks the whole process tree, so the measurement child is
picked up as a grandchild exactly as a child would be.

This is the same shape ``harness`` uses for R, for the same reason, and both
sit on the same measurement spine in ``measure``. What differs is only that the
child here is long-lived and answers requests, rather than being spawned once.
"""
import argparse
import os
import sys
import time

from jumper_extension.adapters.ai_reviewer.benchmark.measure import (
    build_silent_adapter,
    measure_session,
)
from jumper_extension.adapters.ai_reviewer.benchmark.models import FAILED
from jumper_extension.adapters.ai_reviewer.benchmark.replay.link import JsonChannel, JsonLink

_ZYGOTE_MODULE = "jumper_extension.adapters.ai_reviewer.benchmark.replay.zygote"

# How long a stopping zygote gets to leave on its own before it is killed.
_STOP_TIMEOUT_S = 10.0


class Supervisor:
    """Samples a zygote's tree, and turns each of its forks into a session."""

    def __init__(
        self,
        prefix_script: str,
        interval: float,
        prefix_count: int,
        language: str,
        work_dir: str,
        link: JsonLink,
    ):
        self.interval = interval
        self.prefix_count = prefix_count
        self.language = language
        self.work_dir = work_dir
        self.link = link
        self.adapter = build_silent_adapter("supervisor")
        self.zygote = JsonChannel(
            command=[
                sys.executable,
                "-m",
                _ZYGOTE_MODULE,
                "--prefix-script",
                prefix_script,
                "--work-dir",
                work_dir,
            ],
            fd_flag="--response-fd",
            log=open(os.path.join(work_dir, "zygote.log"), "w"),
            cwd=work_dir,
        )

    def start(self) -> dict:
        """Wait for the zygote to build the prefix and vet the machine."""
        ready = self.zygote.receive()
        if ready is None:
            return {"ok": False, "reason": "the zygote exited before it was ready"}
        return ready

    def serve(self):
        """Measure a cell per request until the strategy stops asking."""
        for request in self.link.requests_until_closed():
            self.link.answer(self.measure(request))
        self.zygote.close(_STOP_TIMEOUT_S)

    def measure(self, request: dict) -> dict:
        """Sample the tree while the zygote forks a child to run one cell."""
        if not self.zygote.alive():
            return {"status": FAILED, "error": "the zygote is gone", "gone": True}

        with open(request["code_path"], encoding="utf-8") as handle:
            target_code = handle.read()

        answer: dict = {}
        started = time.perf_counter()

        def run() -> dict:
            reply = self.zygote.ask({**request, "cmd": "fork"})
            if reply is None:
                return {
                    "status": FAILED,
                    "error": "the zygote stopped answering mid-measurement",
                    "gone": True,
                }
            answer.update(reply)
            return reply

        outcome = measure_session(
            adapter=self.adapter,
            interval=self.interval,
            run=run,
            session_path=request["session_path"],
            markers_path=request["markers_path"],
            prefix_count=self.prefix_count,
            target_code=target_code,
            language=self.language,
        )
        return {**outcome, "wall_s": round(time.perf_counter() - started, 4)}

    def close(self):
        self.zygote.close(_STOP_TIMEOUT_S)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JUmPER benchmark fork supervisor")
    parser.add_argument("--prefix-script", required=True)
    parser.add_argument("--interval", type=float, required=True)
    parser.add_argument("--prefix-count", type=int, required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--language", default="python")
    parser.add_argument(
        "--response-fd",
        type=int,
        required=True,
        help="inherited pipe the strategy reads answers from; never stdout",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    supervisor = Supervisor(
        prefix_script=args.prefix_script,
        interval=args.interval,
        prefix_count=args.prefix_count,
        language=args.language,
        work_dir=args.work_dir,
        link=JsonLink(sys.stdin, args.response_fd),
    )
    ready = supervisor.start()
    supervisor.link.answer(ready)
    if not ready.get("ok"):
        supervisor.close()
        return 0  # the reason is the answer; the caller falls back on its own
    supervisor.serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
