"""The plumbing under the fork mode: the answer channel and process lifetimes.

Deterministic on purpose. The end-to-end fork tests can only run where the
fidelity probe accepts the machine, which is exactly the environment - a shared,
loaded CI box - where these failure modes are most likely. These use throwaway
scripts instead, so they run everywhere.
"""
import json
import os
import signal
import sys
import textwrap
import time

import pytest

from jumper_extension.adapters.ai_reviewer.benchmark.replay.lifetime import (
    die_with_parent,
    own_process_group,
)
from jumper_extension.adapters.ai_reviewer.benchmark.replay.link import JsonChannel

posix_only = pytest.mark.skipif(os.name != "posix", reason="the fork mode needs POSIX")


def _script(tmp_path, name: str, body: str) -> str:
    """Write a throwaway far end that answers on the descriptor it was handed."""
    path = tmp_path / name
    path.write_text(textwrap.dedent(_PREAMBLE) + textwrap.dedent(body), encoding="utf-8")
    return str(path)


def _channel(tmp_path, script: str) -> JsonChannel:
    return JsonChannel(
        command=[sys.executable, script],
        fd_flag="--response-fd",
        log=open(tmp_path / "log.txt", "w"),
        cwd=str(tmp_path),
    )


_PREAMBLE = """
    import os, sys
    fd = int(sys.argv[sys.argv.index("--response-fd") + 1])
"""


def test_channel_carries_an_answer_back(tmp_path):
    script = _script(tmp_path, "echo.py", """
        with os.fdopen(fd, "w") as out:
            out.write('{"ok": true, "detail": "hello"}\\n')
            out.flush()
    """)
    channel = _channel(tmp_path, script)

    try:
        assert channel.receive(deadline=time.monotonic() + 30) == {
            "ok": True,
            "detail": "hello",
        }
    finally:
        channel.close(5.0)


@posix_only
def test_channel_notices_a_far_end_that_died_holding_the_pipe_open(tmp_path):
    """The failure that made this a poll rather than a blocking read.

    A prefix that starts worker processes leaves copies of the answer pipe all
    over the machine, so it never reaches end-of-file when its owner dies.
    Waiting for that would wait forever - which hung the notebook.
    """
    script = _script(tmp_path, "holder.py", """
        import time
        if os.fork() == 0:
            time.sleep(60)      # a grandchild keeps the write end open
            os._exit(0)
        time.sleep(0.5)
        os._exit(7)             # the far end leaves without closing anything
    """)
    channel = _channel(tmp_path, script)

    try:
        started = time.perf_counter()
        answer = channel.receive(deadline=time.monotonic() + 30)
        elapsed = time.perf_counter() - started
    finally:
        channel.close(5.0)

    assert answer is None
    assert "exited with status 7" in channel.failure()
    assert elapsed < 20, "a dead far end must be noticed without waiting out the deadline"


def test_channel_gives_up_at_its_deadline(tmp_path):
    script = _script(tmp_path, "mute.py", """
        import time
        time.sleep(60)          # alive, and never answering
    """)
    channel = _channel(tmp_path, script)

    try:
        answer = channel.receive(deadline=time.monotonic() + 1.0)
    finally:
        channel.close(5.0)

    assert answer is None
    assert "stopped answering" in channel.failure()


def test_channel_reports_an_answer_it_cannot_read(tmp_path):
    script = _script(tmp_path, "garbage.py", """
        with os.fdopen(fd, "w") as out:
            out.write("not json at all\\n")
            out.flush()
    """)
    channel = _channel(tmp_path, script)

    try:
        assert channel.receive(deadline=time.monotonic() + 30) is None
    finally:
        channel.close(5.0)
    assert "not an answer" in channel.failure()


def test_closing_the_channel_ends_the_far_end(tmp_path):
    script = _script(tmp_path, "sleeper.py", """
        import time
        time.sleep(60)
    """)
    channel = _channel(tmp_path, script)
    process = channel.process

    channel.close(5.0)

    assert process.poll() is not None


@posix_only
def test_a_child_can_be_put_in_a_group_of_its_own(tmp_path):
    """Run in a fork: moving the test runner's own group would be rude."""
    result = tmp_path / "group.json"
    pid = os.fork()
    if pid == 0:
        placed = own_process_group()
        result.write_text(json.dumps({"placed": placed, "pgid": os.getpgid(0), "pid": os.getpid()}))
        os._exit(0)
    os.waitpid(pid, 0)

    report = json.loads(result.read_text())
    assert report["placed"]
    assert report["pgid"] == report["pid"], "a leader's group id is its own pid"


@posix_only
def test_a_child_asks_to_die_with_its_parent(tmp_path):
    """The guarantee that keeps an abandoned measurement from outliving its zygote."""
    result = tmp_path / "pdeath.json"
    pid = os.fork()
    if pid == 0:
        result.write_text(json.dumps({"armed": die_with_parent(signal.SIGKILL)}))
        os._exit(0)
    os.waitpid(pid, 0)

    assert json.loads(result.read_text())["armed"]
