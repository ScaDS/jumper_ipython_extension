"""Which magics a generated script can carry, and what to do with the rest.

A replay is a plain Python file handed to ``python``, not a cell handed to a
kernel: nothing there understands ``%``. The wrapper kernel solves this at
execution time by forwarding a foreign cell to the wrapped kernel, but that route
does not exist when a script is written to disk and run in a fresh process, so
every magic has to be resolved here first - rewritten, removed, or refused.

The names live in this module rather than being read off ``PerfmonitorMagics``
because that class reaches the script writer through the service and importing it
from there would close an import cycle. ``tests/test_script_rendering.py`` asserts
the two agree, so the duplication cannot drift silently.
"""

# The magics a script can route to ``magic_adapter``, because the adapter has a
# method of that name. Kept in sync with ``PerfmonitorMagics.magics["line"]``.
JUMPER_LINE_MAGICS = frozenset(
    {
        "export_session",
        "import_session",
        "perfmonitor_ai_review",
        "perfmonitor_disable_perfreports",
        "perfmonitor_enable_perfreports",
        "perfmonitor_export_cell_history",
        "perfmonitor_export_perfdata",
        "perfmonitor_fast_setup",
        "perfmonitor_help",
        "perfmonitor_load_cell_history",
        "perfmonitor_load_perfdata",
        "perfmonitor_perfreport",
        "perfmonitor_plot",
        "perfmonitor_resources",
        "perfmonitor_start",
        "perfmonitor_stop",
        "show_cell_history",
        "start_write_script",
        "end_write_script",
    }
)

# Cell magics whose body *is* Python and which only decorate how it runs. The
# header comes off and the body stays, which is what a replay wants: it is timing
# the cell itself, and a second timer or a suppressed stdout would change nothing
# about the state the cell builds.
TRANSPARENT_CELL_MAGICS = frozenset({"time", "timeit", "prun", "debug", "capture"})

# Why each of the rest cannot be rendered. Not an exhaustive list of cell magics -
# anything absent from ``TRANSPARENT_CELL_MAGICS`` is refused anyway - but a named
# reason is worth more to a user than "unsupported".
_BODY_IS_NOT_PYTHON = "the cell body is not Python"
CELL_MAGIC_REASONS = {
    "bash": _BODY_IS_NOT_PYTHON,
    "sh": _BODY_IS_NOT_PYTHON,
    "script": _BODY_IS_NOT_PYTHON,
    "perl": _BODY_IS_NOT_PYTHON,
    "ruby": _BODY_IS_NOT_PYTHON,
    "html": _BODY_IS_NOT_PYTHON,
    "javascript": _BODY_IS_NOT_PYTHON,
    "js": _BODY_IS_NOT_PYTHON,
    "latex": _BODY_IS_NOT_PYTHON,
    "markdown": _BODY_IS_NOT_PYTHON,
    "svg": _BODY_IS_NOT_PYTHON,
    "writefile": "it writes the body to a file rather than running it",
    "file": "it writes the body to a file rather than running it",
}


def cell_magic_reason(name: str, arguments: str) -> str:
    """Why ``%%name arguments`` cannot be rendered, or "" when it can.

    ``%%capture`` is the one that depends on its arguments: bare, it only
    suppresses output and the body can run as it is; given a name, it binds that
    name to the captured output, and a later cell reading it would fail on a
    variable the replay never created.
    """
    if name == "capture" and arguments.strip():
        return f"it binds the cell's output to {arguments.strip()!r}, which a replay cannot produce"
    if name in TRANSPARENT_CELL_MAGICS:
        return ""
    return CELL_MAGIC_REASONS.get(name, "a replay runs a plain Python script, and nothing there reads it")
