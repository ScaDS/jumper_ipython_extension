import logging
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import Optional, List
from datetime import datetime

from jumper_extension.core.state import State
from jumper_extension.adapters.cell_history import CellHistory
from jumper_extension.adapters.magic_names import (
    JUMPER_LINE_MAGICS,
    TRANSPARENT_LINE_MAGICS,
    cell_magic_reason,
)

logger = logging.getLogger("extension")


def is_pure_magic_cell(raw_cell: str) -> bool:
    """True when every non-empty line of *raw_cell* is a line magic."""
    non_empty_lines = [line for line in raw_cell.splitlines() if line.strip()]
    return bool(non_empty_lines) and all(
        line.lstrip().startswith("%") for line in non_empty_lines
    )


@dataclass
class RenderedCell:
    """One notebook cell as script source, and what did not survive the trip.

    *unsupported* is the important field: it is empty when *source* really is the
    cell, and a reason when the cell cannot be a Python script at all. A caller
    that ignores it gets what the benchmark used to get - a file that does not
    parse, blamed on whoever wrote the cell.
    """
    source: str
    dropped: List[str] = field(default_factory=list)
    unsupported: str = ""


def render_source(raw_cell: str, cell_magics: List[str]) -> RenderedCell:
    """Turn one cell into source a plain interpreter can run.

    Three kinds of magic reach this function and each leaves differently:

    - a JUmPER magic becomes a ``magic_adapter`` call, because the adapter has a
      method of that name and the replay wants the effect;
    - any other magic is **removed**, because nothing in a script can serve it -
      there is no frontend for ``%matplotlib`` and no autoreloader to configure.
      It used to be rewritten to ``magic_adapter.matplotlib(...)`` and fail with
      ``AttributeError``;
    - a cell magic that decorates Python (``%%time``) loses its header and keeps
      its body; one whose body is not Python (``%%bash``) is refused outright.

    **The returned source parses, or *unsupported* says why it cannot.** That is
    the guarantee callers need, and it cannot be met from the captured magic list
    alone: ``cell_magics`` holds what IPython recognised *before* the cell ran, so
    a cell that begins ``%load_ext autoreload`` and then uses ``%autoreload 2``
    reports only the first - the second magic did not exist yet. What is left is
    therefore removed by :func:`_drop_unparseable_lines`, which asks the parser
    instead of guessing. A ``%`` inside a string literal never breaks parsing and
    is never touched.
    """
    if not raw_cell:
        return RenderedCell("")

    body, dropped, unsupported = _strip_cell_magic(raw_cell)
    if unsupported:
        return RenderedCell("", dropped, unsupported)

    replacements, foreign = _adapter_calls(cell_magics)
    out_lines: List[str] = []
    for line in body.splitlines():
        lstrip = line.lstrip()
        # Only attempt replacement if line starts with a magic marker
        if lstrip.startswith("%"):
            indent = line[: len(line) - len(lstrip)]
            name = _magic_name(lstrip)
            if name in foreign:
                dropped.append(lstrip)
                # `pass`, not a bare comment: this line may be the only thing in
                # a block, and an empty block is a fresh syntax error one line up.
                out_lines.append(
                    f"{indent}pass  # JUmPER dropped {lstrip.strip()}: {foreign[name]}"
                )
                continue
            rep = _lookup(lstrip, replacements)
            if rep is not None:
                # keep original indentation
                out_lines.append(f"{indent}{rep}")
                continue
        out_lines.append(line)

    source, swept = _drop_unparseable_lines("\n".join(out_lines))
    return RenderedCell(source, dropped + swept, "")


def _drop_unparseable_lines(source: str) -> tuple:
    """Remove the magic and shell lines the parser trips over, and only those.

    Everything IPython accepts that Python does not - ``%magic``, ``!command``,
    a trailing ``?`` - is a syntax error at exactly its own line, so the parser
    is a better detector than any prefix rule: it finds a magic nobody captured
    and leaves alone a ``%`` that merely appears inside a string, because that
    one parses.

    Lines are replaced by a comment rather than deleted so the line numbers the
    next parse reports still refer to the same places, and so the script shows
    what was taken out. A syntax error that is not one of these is the user's
    own, and is handed back untouched for the replay to report.
    """
    lines = source.splitlines()
    dropped: List[str] = []
    for _ in range(len(lines) + 1):
        try:
            compile("\n".join(lines), "<cell>", "exec")
            return "\n".join(lines), dropped
        except SyntaxError as error:
            index = (error.lineno or 0) - 1
            if not 0 <= index < len(lines):
                break
            offender = lines[index]
            stripped = offender.strip()
            if not stripped.startswith(("%", "!")) and not stripped.endswith("?"):
                break
            indent = offender[: len(offender) - len(offender.lstrip())]
            dropped.append(stripped)
            lines[index] = indent + _without_magic(stripped)
        except ValueError:
            # A null byte or similar: nothing here can help, and the replay's own
            # error is more informative than a mangled source would be.
            break
    # Nothing was salvaged, so hand back exactly what came in: the replay's own
    # report of the user's syntax error is worth more than a half-edited cell.
    return source, []


def _without_magic(stripped_line: str) -> str:
    """What a line that Python cannot parse leaves behind.

    ``%time total = f()`` keeps its assignment - the magic decorated a statement,
    and the statement is the part a replay has to reproduce. Everything else
    becomes ``pass``, not a bare comment: the line may be the only thing in a
    block, and a block with nothing in it is a fresh syntax error one line up.
    """
    note = f"# JUmPER dropped {stripped_line}"
    if stripped_line.startswith("%"):
        parts = stripped_line.lstrip("%").split(maxsplit=1)
        name = parts[0] if parts else ""
        remainder = parts[1].strip() if len(parts) > 1 else ""
        if name in TRANSPARENT_LINE_MAGICS and remainder:
            return f"{remainder}  {note}: kept the statement, lost the timing"
    return f"pass  {note}: not Python, and no kernel to run it"


def transform_cell_code(raw_cell: str, cell_magics: List[str]) -> str:
    """Replace captured magic commands with magic_adapter calls.

    The source of :func:`render_source`, for callers that have already decided
    what to do about a cell they cannot render.
    """
    return render_source(raw_cell, cell_magics).source


def _strip_cell_magic(raw_cell: str) -> tuple:
    """Take a ``%%`` header off a cell, or say why the cell cannot be rendered.

    A cell magic has to be the cell's first non-blank line - that is IPython's
    rule, not ours - so only that line is examined.
    """
    lines = raw_cell.splitlines()
    first = next((index for index, line in enumerate(lines) if line.strip()), None)
    if first is None or not lines[first].lstrip().startswith("%%"):
        return raw_cell, [], ""

    header = lines[first].strip()
    parts = header[2:].split(maxsplit=1)
    name = parts[0] if parts else ""
    arguments = parts[1] if len(parts) > 1 else ""
    reason = cell_magic_reason(name, arguments)
    if reason:
        return "", [], f"{header.split()[0]}: {reason}"

    remaining = lines[:first] + lines[first + 1:]
    return "\n".join(remaining), [header], ""


def _adapter_calls(cell_magics: List[str]) -> tuple:
    """Split the captured magics into ones the adapter serves and ones it cannot.

    Returns the replacement lookup for the first group and, for the second, the
    reason each is dropped - keyed by name, since that is what a line is matched
    on once the exact-text lookup misses.
    """
    replacements: dict = {}
    foreign: dict = {}
    for magic in cell_magics:
        # Normalize leading '%'
        stripped_no_pct = magic[1:] if magic.startswith("%") else magic
        parts = stripped_no_pct.split(maxsplit=1)
        if not parts:
            continue
        cmd = parts[0]
        args = parts[1] if len(parts) > 1 else ""
        if cmd not in JUMPER_LINE_MAGICS:
            foreign[cmd] = "no notebook to serve it outside a live session"
            continue
        # construct a Python call to the magic_adapter method
        # prefer passing the whole "line" string of arguments
        if args:
            call = f"magic_adapter.{cmd}({args!r})"
        else:
            # Methods generally accept a single 'line' argument; pass empty string for uniformity
            call = f'magic_adapter.{cmd}("")'
        # map original magic literal (with or without %) to replacement
        replacements[magic] = call
        # also allow matching without the leading '%', just in case
        replacements[stripped_no_pct] = call
    return replacements, foreign


def _magic_name(lstripped_line: str) -> str:
    """The command a magic line invokes, without its ``%`` or arguments."""
    tokens = lstripped_line.lstrip("%").split(maxsplit=1)
    return tokens[0] if tokens else ""


def _lookup(lstripped_line: str, replacements: dict) -> Optional[str]:
    """The adapter call for a magic line, matched the three ways it may appear."""
    # Exact match by full line (common for bare magic lines)
    rep = replacements.get(lstripped_line)
    if rep is not None:
        return rep
    # Try by the first token
    key = lstripped_line.split("#", 1)[0].strip()  # drop trailing inline comments if any
    rep = replacements.get(key)
    if rep is not None:
        return rep
    # As a fallback, try to parse and replace if it's one of captured commands
    token = _magic_name(lstripped_line)
    for captured, call in replacements.items():
        if captured.lstrip("%").split(maxsplit=1)[0] == token:
            return call
    return None


class NotebookScriptWriter:
    """
    Class for writing notebook content to a Python script.

    Collects code from cells and saves it to a Python file with optional
    metadata about execution time and cell numbers.
    """

    def __init__(self, cell_history: CellHistory, default_interval: float):
        self.cell_history = cell_history
        self.output_path = None
        # recording state
        self._recording = False
        self._start_time = None
        self._start_cell_index: Optional[int] = None
        self._state = State()
        self._default_interval = default_interval
        # names of magics that start/stop script writing (to exclude their cells)
        self._control_magics = {"start_write_script", "end_write_script"}

    def is_recording_active(self) -> bool:
        """Check if cell is being recorded."""
        return self._recording

    def start_recording(self, state: State, output_path: Optional[str] = None):
        """
        Start recording code from cells.

        Args:
            state: Extension runtime state at the time recording started
            output_path: Path to the output file (overrides value from __init__)
        """
        self._state = state
        if output_path:
            self.output_path = output_path
        else:
            # Generate default filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = f"notebook_script_{timestamp}.py"

        # mark the current tail of CellHistory; everything after this point is "to be written"
        self._recording = True
        self._start_time = datetime.now()
        # exclude the cell that triggered the start magic itself
        self._start_cell_index = len(self.cell_history)

    def stop_recording(self) -> Optional[str]:
        """
        Stop recording and save accumulated code to file.

        Returns:
            Path to the created file or None on error
        """
        if not self._recording:
            logger.warning("[JUmPER]: Recording was not started")
            return None

        # collect cells recorded since start, excluding start/end control magic cells
        try:
            history = self.cell_history.view()
        except Exception as e:
            logger.error(f"[JUmPER]: Failed to access CellHistory: {e}")
            return None

        if history is None or history.empty:
            logger.warning("[JUmPER]: No cells in CellHistory")
            return None

        selected = []
        for _, row in history.iterrows():
            try:
                idx = int(row.get("cell_index"))
            except Exception:
                continue
            if self._start_cell_index is not None and idx < self._start_cell_index:
                continue
            if self.is_control_cell(row.get("cell_magics")):
                continue
            selected.append(
                {
                    "index": idx,
                    "timestamp": datetime.fromtimestamp(row["start_time"])
                    if isinstance(row.get("start_time"), (int, float))
                    else self._start_time or datetime.now(),
                    "raw_cell": row.get("raw_cell", ""),
                    "cell_magics": row.get("cell_magics") or [],
                }
            )

        if not selected:
            logger.warning("[JUmPER]: No recorded cells to save")
            # reset state
            self._recording = False
            self._start_cell_index = None
            return None

        try:
            self._write_to_file(selected)
            logger.info(
                f"[JUmPER]: Recorded {len(selected)} cells "
                f"to file '{self.output_path}'"
            )
            return self.output_path
        except Exception as e:
            logger.error(
                f"[JUmPER]: Error writing file: {e}"
            )
            return None
        finally:
            # reset state
            self._recording = False
            self._start_cell_index = None

    def is_control_cell(self, cell_magics):
        """Select cells with index >= start and exclude cells that contain control magics"""
        if cell_magics is None:
            return False
        try:
            for m in cell_magics:
                # m may be like "%perfmonitor_start ..." or "perfmonitor_start ..."
                name = m.lstrip("%")
                name = name.split(maxsplit=1)[0]
                if name in self._control_magics:
                    return True
        except Exception:
            pass
        return False
    def _write_to_file(self, recorded_cells: List[dict]):
        """
        Write accumulated cells to Python file.
        """
        output_path = Path(self.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            # File header
            header = dedent(f"""\
                #!/usr/bin/env python3
                \"\"\"
                Auto-generated script from Jupyter notebook
                Generated: {datetime.now():%Y-%m-%d %H:%M:%S}
                Recording started: {self._start_time:%Y-%m-%d %H:%M:%S} if self._start_time else ""
                Total cells: {len(recorded_cells)}
                \"\"\"

                from jumper_extension.core.service import build_perfmonitor_magic_adapter
                magic_adapter = build_perfmonitor_magic_adapter(
                    plots_disabled=True,
                    plots_disabled_reason="Plotting disabled in generated script.",
                    display_disabled=True,
                    display_disabled_reason="Display disabled in generated script."
                )
                
                {self._restore_perfmonitor()}
            """)
            f.write(header)

            # Write code from each recorded cell
            for cell in recorded_cells:
                f.write(f"# Cell {cell['index']}\n")
                ts = cell['timestamp']
                f.write(f"# Recorded at: {ts.strftime('%H:%M:%S') if isinstance(ts, datetime) else ts}\n")
                raw_cell = cell.get("raw_cell", "")
                f.write("# --- Cell print ---\n")
                f.write(f"raw_cell = {raw_cell!r}\n")
                f.write(f"print('-' * 40)\n")
                f.write(f"print('Cell {cell['index']}')\n")
                f.write(f"print('-' * 40)\n")
                f.write(f"print(raw_cell)\n")
                f.write("print('-' * 13 + ' Cell output ' + '-' * 14)\n")
                cell_magics = cell.get("cell_magics") or []
                # compute should_skip_report: True if cell contains only line magics (non-empty lines start with '%')
                is_pure_magic = is_pure_magic_cell(raw_cell)
                f.write(
                    "magic_adapter.on_pre_run_cell("
                    f"raw_cell, "
                    f"{cell_magics!r}, "
                    f"{is_pure_magic!r}"
                    ")\n"
                )
                f.write("# --- Cell content ---\n")
                transformed = transform_cell_code(
                    raw_cell,
                    cell_magics
                )
                f.write(f"{transformed}\n")
                f.write("# --- Cell End -------\n")
                f.write("magic_adapter.on_post_run_cell('')\n")
                f.write("\n")
            base_name = output_path.stem
            perf_csv = f"{base_name}_perfdata.csv"
            cell_csv = f"{base_name}_cell_history.csv"
            footer = dedent(
                f"""\
                # --- Export results to CSV ---
                # Performance data by level (default level from settings)
                magic_adapter.perfmonitor_export_perfdata("--file {perf_csv}")
                # Cell execution history
                magic_adapter.perfmonitor_export_cell_history("--file {cell_csv}")
                """
            )
            f.write(footer)

    def _restore_perfmonitor(self) -> str:
        if self._state.monitoring.running:
            state = self._state

            # Determine interval to restore
            interval = state.monitoring.user_interval
            if not interval:
                interval = self._default_interval

            # If auto-reports were enabled, a single enable call will both start monitoring
            # (if needed) and configure reports consistently with original settings.
            if state.perfreports.enabled:
                level = state.perfreports.level
                args = f"--level {level} --interval {interval}"
                if state.perfreports.text:
                    args += " --text"
                return f"magic_adapter.perfmonitor_enable_perfreports({args!r})\n"

            # Otherwise just restore monitor start with the same interval
            return f"magic_adapter.perfmonitor_start({str(interval)!r})\n"

        return ""
