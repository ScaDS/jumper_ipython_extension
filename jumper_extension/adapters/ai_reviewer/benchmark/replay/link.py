"""One request, one answer, one line of JSON each - over a pipe of its own.

The fork mode is a chain of processes, so this channel is spoken at several ends
and is worth having in one place. Three rules make it safe, and all three were
learned the hard way:

* **Never stdout.** The extension logs a line to stdout the moment it is
  imported, and a replayed cell can print anything at all. A channel sharing
  that stream is corrupted before its owner gets control, so answers travel on a
  dedicated inherited descriptor.
* **Flush every answer.** A buffered half-message inherited across a fork would
  be written out a second time by the child.
* **Silence is not proof of life, and an open pipe is not either.** A pipe stays
  open as long as *any* process holds its write end, and a prefix that started
  worker processes leaves copies of it all over the machine. Waiting for
  end-of-file would then wait forever behind a far end that is already dead, so
  the asking end watches the process itself instead, and can be given a deadline
  on top.

``receive`` returning None always means the same thing - no answer is coming -
and ``failure()`` says which of the ways it was.
"""
import json
import os
import select
import subprocess
import time

# How often the asking end looks up from the pipe to check the far end is still
# alive. Short enough to notice promptly, long enough to cost nothing.
_LIVENESS_TICK_S = 0.25

_READ_CHUNK = 65536


class JsonLink:
    """The answering end of the channel: reads requests, writes answers."""

    def __init__(self, requests, response_fd: int):
        self._requests = requests
        # Not inherited across exec: a cell that shells out has no business
        # holding the channel its own supervisor is waiting on.
        os.set_inheritable(response_fd, False)
        self._responses = os.fdopen(response_fd, "w")

    def requests_until_closed(self):
        """Yield each request until the far end stops asking or says to stop."""
        while True:
            line = self._requests.readline()
            if not line:
                return
            line = line.strip()
            if not line:
                continue
            request = json.loads(line)
            if request.get("cmd") == "stop":
                return
            yield request

    def answer(self, payload: dict):
        self._responses.write(json.dumps(payload) + "\n")
        self._responses.flush()

    def close(self):
        try:
            self._responses.close()
        except OSError:
            pass


class JsonChannel:
    """The asking end: owns a child process and talks to it.

    The child is spawned with an extra descriptor to answer on, and the parent
    drops its own copy of the write end immediately - otherwise the parent would
    be holding open the very pipe it uses to notice the child is gone.
    """

    def __init__(self, command: list[str], fd_flag: str, log, **popen_kwargs):
        read_fd, write_fd = os.pipe()
        try:
            self.process = subprocess.Popen(
                [*command, fd_flag, str(write_fd)],
                stdin=subprocess.PIPE,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                pass_fds=(write_fd,),
                **popen_kwargs,
            )
        finally:
            os.close(write_fd)
        self._read_fd = read_fd
        self._buffer = b""
        self._failure = ""

    def alive(self) -> bool:
        return self.process.poll() is None

    def failure(self) -> str:
        """Why the last ``receive`` came back empty."""
        return self._failure

    def ask(self, request: dict, deadline: float | None = None) -> dict | None:
        """Send *request* and read its answer; None once no answer can come."""
        try:
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            self._failure = "the channel closed before the request could be sent"
            return None
        return self.receive(deadline)

    def receive(self, deadline: float | None = None) -> dict | None:
        """One answer, or None when the far end died, closed, or ran out of time.

        *deadline* is a ``time.monotonic`` value. Leaving it unset waits as long
        as the far end lives, which is right for work whose length is the user's
        to decide - a prefix that takes ten minutes is slow, not broken.
        """
        line = self._read_line(deadline)
        if line is None:
            return None
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            self._failure = "the far end sent something that was not an answer"
            return None

    def _read_line(self, deadline: float | None) -> bytes | None:
        """Raw reads rather than a buffered stream, so the poll below is the truth."""
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                line, self._buffer = self._buffer[:newline], self._buffer[newline + 1:]
                return line

            ready, _, _ = select.select([self._read_fd], [], [], _LIVENESS_TICK_S)
            if ready:
                try:
                    chunk = os.read(self._read_fd, _READ_CHUNK)
                except OSError:
                    chunk = b""
                if chunk:
                    self._buffer += chunk
                    continue
                self._failure = "the far end closed the channel"
                return None

            if self.process.poll() is not None:
                self._failure = f"the far end exited with status {self.process.returncode}"
                return None
            if deadline is not None and time.monotonic() > deadline:
                self._failure = "the far end stopped answering"
                return None

    def close(self, stop_timeout: float):
        """Ask the child to leave, then make sure it did."""
        process = self.process
        try:
            if process.poll() is None:
                process.stdin.write(json.dumps({"cmd": "stop"}) + "\n")
                process.stdin.flush()
                process.wait(timeout=stop_timeout)
        except Exception:
            pass
        finally:
            if process.poll() is None:
                process.kill()
                try:
                    process.wait(timeout=stop_timeout)
                except Exception:
                    pass
            try:
                if process.stdin is not None:
                    process.stdin.close()
            except OSError:
                pass
            try:
                os.close(self._read_fd)
            except OSError:
                pass
