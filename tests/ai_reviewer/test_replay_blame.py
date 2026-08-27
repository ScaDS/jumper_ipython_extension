"""Who a failed replay is reported against.

A benchmark replays the prefix and then times one cell. When the prefix is what
failed, the cell under review never ran - so reporting the failure as *its*
failure points the user, and the repair loop, at code that is very likely fine.
These check the two are told apart.
"""
from pathlib import Path

from jumper_extension.adapters.ai_reviewer.benchmark.models import RunOutcome
from jumper_extension.adapters.ai_reviewer.benchmark.orchestrator import _baseline_failure
from jumper_extension.adapters.ai_reviewer.benchmark.script import (
    body_path_for,
    build_script,
    failing_checkpoint_cell,
    failing_prefix_cell,
)

_PREFIX = [
    {"index": 0, "raw_cell": "import os", "cell_magics": []},
    {"index": 3, "raw_cell": "hv.notebook_extension('bokeh')", "cell_magics": []},
]


def _script(tmp_path) -> tuple:
    """The entry script, plus the body module the cells and tracebacks live in."""
    path = build_script(
        _PREFIX,
        "total = 1",
        0.05,
        ["total"],
        "s.zip",
        "f.json",
        str(tmp_path / "baseline_0.py"),
    )
    body = body_path_for(path)
    return path, body, open(body).read().splitlines()


def _traceback(path: str, line: int, message: str) -> str:
    return (
        "Traceback (most recent call last):\n"
        f'  File "{path}", line {line}, in <module>\n'
        f"{message}"
    )


def test_a_failure_in_the_prefix_names_the_cell_it_came_from(tmp_path):
    path, body, lines = _script(tmp_path)
    # The rendered call, not the hook that quotes the same text back.
    failing = next(
        number
        for number, line in enumerate(lines, start=1)
        if line.startswith("hv.notebook_extension")
    )

    assert failing_prefix_cell(path, _traceback(body, failing, "RuntimeError: no notebook")) == 3


def test_a_variant_script_is_importable_whatever_the_tag(tmp_path):
    """Variant tags start with a digit, and `import 1_0__body` is a syntax error."""
    path = build_script(
        _PREFIX,
        "total = 1",
        0.05,
        ["total"],
        "s.zip",
        "f.json",
        str(tmp_path / "1_0.py"),
    )

    compile(open(path).read(), path, "exec")
    assert Path(body_path_for(path)).stem.isidentifier()


def test_a_failure_in_the_cell_under_test_is_not_blamed_on_the_prefix(tmp_path):
    path, body, lines = _script(tmp_path)
    failing = next(
        number for number, line in enumerate(lines, start=1) if line.strip() == "total = 1"
    )

    assert failing_prefix_cell(path, _traceback(body, failing, "ValueError: boom")) is None


def test_an_error_from_somewhere_else_resolves_to_nothing(tmp_path):
    path, _, _ = _script(tmp_path)

    assert failing_prefix_cell(path, "ImportError: no module named minian") is None
    assert failing_prefix_cell("/tmp/does-not-exist.py", 'File "/tmp/x.py", line 2') is None


def test_the_message_says_the_review_is_unaffected():
    outcome = RunOutcome(
        status="failed",
        error='  File "/tmp/b.py", line 22\nRuntimeError: Jupyter notebook not available: '
        "use hv.extension instead.",
        prefix_cell=3,
    )

    message = _baseline_failure(outcome)

    assert "benchmark skipped: prefix cell 3 could not be replayed" in message
    assert "RuntimeError: Jupyter notebook not available" in message
    assert "The review itself is unaffected." in message


def test_without_a_prefix_cell_the_message_is_the_one_about_the_reviewed_cell():
    message = _baseline_failure(RunOutcome(status="failed", error="ValueError: boom"))

    assert "the cell under review did not replay cleanly" in message
    assert "prefix cell" not in message


def test_a_checkpoint_traceback_names_its_cell():
    # The dill checkpoint hands each cell to run_cell, which compiles it as
    # `<cell N>` - there is no file to map line numbers against.
    error = 'File "<cell 3>", line 2, in <module>\nRuntimeError: no notebook'

    assert failing_checkpoint_cell(error) == 3
    assert failing_checkpoint_cell("RuntimeError: no notebook") is None
