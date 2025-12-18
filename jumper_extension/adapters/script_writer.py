import logging
from pathlib import Path
from textwrap import dedent
from typing import Optional, List
from datetime import datetime

from jumper_extension.core.state import Settings
from jumper_extension.adapters.cell_history import CellHistory

logger = logging.getLogger("extension")


class NotebookScriptWriter:
    """
    Class for writing notebook content to a Python script.

    Collects code from cells and saves it to a Python file with optional
    metadata about execution time and cell numbers.
    """

    def __init__(self, cell_history: CellHistory):
        self.cell_history = cell_history
        self.output_path = None
        # recording state
        self._recording = False
        self._start_time = None
        self._start_cell_index: Optional[int] = None
        self._settings_state = Settings()
        # names of magics that start/stop script writing (to exclude their cells)
        self._control_magics = {"start_write_script", "end_write_script"}

    def start_recording(self, settings_state: Settings, output_path: Optional[str] = None):
        """
        Start recording code from cells.

        Args:
            settings_state: Extension settings at the time of recording started
            output_path: Path to the output file (overrides value from __init__)
        """
        self._settings_state = settings_state
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

        logger.info(
            f"[JUmPER]: Started recording to file '{self.output_path}'"
        )

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
            if self._is_control_cell(row.get("cell_magics")):
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

    def _is_control_cell(self, cell_magics):
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
                non_empty_lines = [ln for ln in raw_cell.splitlines() if ln.strip()]
                is_pure_magic = bool(non_empty_lines) and all(ln.lstrip().startswith("%") for ln in non_empty_lines)
                f.write(
                    "magic_adapter.on_pre_run_cell("
                    f"raw_cell, "
                    f"{cell_magics!r}, "
                    f"{is_pure_magic!r}"
                    ")\n"
                )
                f.write("# --- Cell content ---\n")
                transformed = self._transform_cell_code(
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
        if self._settings_state.monitoring.running:
            settings = self._settings_state

            # Determine interval to restore
            interval = settings.monitoring.user_interval
            if not interval:
                interval = settings.monitoring.default_interval

            # If auto-reports were enabled, a single enable call will both start monitoring
            # (if needed) and configure reports consistently with original settings.
            if settings.perfreports.enabled:
                level = settings.perfreports.level
                args = f"--level {level} --interval {interval}"
                if settings.perfreports.text:
                    args += " --text"
                return f"magic_adapter.perfmonitor_enable_perfreports({args!r})\n"

            # Otherwise just restore monitor start with the same interval
            return f"magic_adapter.perfmonitor_start({str(interval)!r})\n"

        return ""

    def _transform_cell_code(self, raw_cell: str, cell_magics: List[str]) -> str:
        """
        Replace captured magic commands with magic_adapter calls while keeping
        the rest of the code intact.
        """
        if not raw_cell:
            return ""

        # Build a lookup from magic line prefix to magic_adapter call string
        # We rely on CellHistory.cell_magics entries having the original magic lines (e.g. "%perfmonitor_start 1.0")
        replacements = {}
        for magic in cell_magics:
            # Normalize leading '%'
            stripped_no_pct = magic[1:] if magic.startswith("%") else magic
            parts = stripped_no_pct.split(maxsplit=1)
            cmd = parts[0]
            args = parts[1] if len(parts) > 1 else ""
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

        # Now transform the cell line by line
        out_lines: List[str] = []
        for line in raw_cell.splitlines():
            lstrip = line.lstrip()
            # Only attempt replacement if line starts with a magic marker
            if lstrip.startswith("%"):
                # Exact match by full line (common for bare magic lines)
                rep = replacements.get(lstrip)
                if rep is None:
                    # Try by the first token
                    key = lstrip.split("#", 1)[0].strip()  # drop trailing inline comments if any
                    rep = replacements.get(key)
                if rep is None:
                    # As a fallback, try to parse and replace if it's one of captured commands
                    token = lstrip[1:].split(maxsplit=1)[0]
                    for k, v in replacements.items():
                        if k.lstrip("%").split(maxsplit=1)[0] == token:
                            rep = v
                            break
                if rep is not None:
                    # keep original indentation
                    indent = line[: len(line) - len(lstrip)]
                    out_lines.append(f"{indent}{rep}")
                    continue
            out_lines.append(line)
        return "\n".join(out_lines)