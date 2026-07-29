"""One request, one answer, one line of JSON each - over a pipe of its own.

The fork mode is a chain of three processes, so this channel is spoken at four
ends and is worth having in one place. Two rules make it safe, and both are
learned rather than obvious:

* **Never stdout.** The extension logs a line to stdout the moment it is
  imported, and a replayed cell can print anything at all. A channel sharing
  that stream is corrupted before its owner gets control, so responses travel on
  a dedicated inherited descriptor.
* **Flush every answer.** A buffered half-message inherited across a fork would
  be written out a second time by the child.

``receive`` returning None always means the same thing: the far end is gone.
"""
import json
import os
import subprocess


class JsonLink:
    """The answering end of the channel: reads requests, writes answers."""

    def __init__(self, requests, response_fd: int):
        self._requests = requests
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
    drops its own copy of the write end immediately - otherwise a dead child
    would never read as end-of-file, and a caller would wait for an answer that
    can no longer come.
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
        self._responses = os.fdopen(read_fd, "r")

    def alive(self) -> bool:
        return self.process.poll() is None

    def ask(self, request: dict) -> dict | None:
        """Send *request* and read its answer; None once the far end is gone."""
        try:
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            return None
        return self.receive()

    def receive(self) -> dict | None:
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
            for stream in (process.stdin, self._responses):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass
