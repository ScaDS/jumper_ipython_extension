"""What survives the trip from a notebook cell to a script a plain interpreter runs.

A replay is a file handed to ``python``, not a cell handed to a kernel, so every
magic has to be resolved before it is written out. These are pure text checks: the
failure they exist to prevent is a generated script that does not parse, and that
is visible without running anything.
"""
import ast

import pytest

from jumper_extension.adapters.magic_names import JUMPER_LINE_MAGICS
from jumper_extension.adapters.script_writer import render_source, transform_cell_code


def parses(source: str) -> bool:
    try:
        ast.parse(source)
    except SyntaxError:
        return False
    return True


def test_a_transparent_cell_magic_loses_its_header_and_keeps_its_body():
    # %%time only decorates how the body runs, and the benchmark is timing the
    # body itself. Left in place it made the whole file unparseable, which was
    # then reported as the reviewed cell failing to replay.
    rendered = render_source("%%time\nimport os\ntotal = sum(range(10))", [])

    assert rendered.unsupported == ""
    assert rendered.source == "import os\ntotal = sum(range(10))"
    assert parses(rendered.source)


@pytest.mark.parametrize("cell", ["%%bash\nmkdir data", "%%writefile out.txt\nhello"])
def test_a_cell_whose_body_is_not_python_is_refused_with_a_reason(cell):
    rendered = render_source(cell, [])

    assert rendered.unsupported.startswith(cell.split()[0])
    assert rendered.source == ""


def test_capture_is_transparent_bare_and_refused_when_it_binds_a_name():
    # Bare, it only suppresses output. Named, a later cell reads a variable this
    # replay would never create, so the failure would surface far from its cause.
    assert render_source("%%capture\nprint(1)", []).unsupported == ""
    assert "out" in render_source("%%capture out\nprint(1)", []).unsupported


def test_a_jumper_magic_is_routed_to_the_adapter():
    rendered = render_source("%perfmonitor_start 0.05\nx = 1", ["perfmonitor_start 0.05"])

    assert "magic_adapter.perfmonitor_start('0.05')" in rendered.source
    assert parses(rendered.source)


def test_a_foreign_magic_is_dropped_rather_than_sent_to_an_adapter_without_it():
    # The adapter has no `matplotlib` method, so rewriting this to a call traded a
    # SyntaxError for an AttributeError. Nothing in a script can serve it either way.
    rendered = render_source("%matplotlib inline\nimport os", ["matplotlib inline"])

    assert "magic_adapter.matplotlib" not in rendered.source
    assert rendered.dropped == ["%matplotlib inline"]
    assert "import os" in rendered.source
    assert parses(rendered.source)


def test_both_kinds_of_magic_in_one_cell():
    rendered = render_source(
        "%matplotlib inline\n%perfmonitor_start 0.05\nx = 1",
        ["matplotlib inline", "perfmonitor_start 0.05"],
    )

    assert "magic_adapter.perfmonitor_start" in rendered.source
    assert "magic_adapter.matplotlib" not in rendered.source
    assert parses(rendered.source)


def test_a_magic_registered_by_an_earlier_line_is_still_removed():
    # The cell that broke a real notebook. `cell_magics` holds what IPython knew
    # *before* the cell ran, so %autoreload - registered by the %load_ext on the
    # line above it - was never captured and went into the script verbatim.
    cell = "%load_ext autoreload\n%autoreload 2\nimport os\nimport sys"

    rendered = render_source(cell, ["load_ext autoreload"])

    assert rendered.unsupported == ""
    assert rendered.dropped == ["%load_ext autoreload", "%autoreload 2"]
    assert "import os" in rendered.source
    assert parses(rendered.source)


@pytest.mark.parametrize(
    "cell",
    [
        "if True:\n    !pip install foo\nx = 1",
        "if True:\n    %matplotlib inline\nx = 1",
        "import os\nos.getcwd?",
        "!ls\nimport os",
    ],
)
def test_everything_the_parser_trips_over_is_removed_safely(cell):
    # Shell escapes, help suffixes and magics inside a block. The last one is why
    # a dropped line becomes `pass` rather than a comment: a block with only a
    # comment in it is a new syntax error one line up.
    magics = ["matplotlib inline"] if "%matplotlib" in cell else []

    assert parses(render_source(cell, magics).source)


def test_a_line_magic_decorating_a_statement_keeps_the_statement():
    # %time binds a name. Dropping the whole line would lose the binding along
    # with the timing, and the next cell would fail on a name that never existed.
    rendered = render_source("%time total = sum(range(10))", [])

    assert rendered.source.startswith("total = sum(range(10))")
    assert parses(rendered.source)


def test_a_syntax_error_of_the_users_own_is_handed_back_untouched():
    # Nothing here can help, and the replay reporting the real error beats a
    # half-edited cell that fails somewhere else.
    cell = "def f(:\n    pass"

    rendered = render_source(cell, [])

    assert rendered.source == cell
    assert rendered.dropped == []


def test_a_percent_inside_a_string_is_left_alone():
    # Only magics IPython captured while the cell ran are removed, so text that
    # merely looks like one keeps its place - and its string keeps its contents.
    cell = 'text = """\n%not a magic\n"""'

    assert render_source(cell, []).source == cell


def test_transform_cell_code_still_returns_source():
    # The old entry point, kept for callers that have already decided what to do
    # about a cell they cannot render.
    assert transform_cell_code("%%time\nx = 1", []) == "x = 1"
    assert transform_cell_code("", []) == ""


def test_the_catalog_matches_the_magics_actually_registered():
    # magic_names.py lists them by hand because reading them off PerfmonitorMagics
    # would close an import cycle through the service. This is what keeps the two
    # from drifting.
    from jumper_extension.ipython.magics import PerfmonitorMagics

    assert JUMPER_LINE_MAGICS == frozenset(PerfmonitorMagics.magics["line"])
