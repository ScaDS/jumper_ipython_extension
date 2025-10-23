import logging
from pathlib import Path
from textwrap import dedent
from typing import Optional, List
from datetime import datetime

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
        # names of magics that start/stop script writing (to exclude their cells)
        self._control_magics = {"start_write_script", "end_write_script"}

    def start_recording(self, output_path: Optional[str] = None):
        """
        Start recording code from cells.

        Args:
            output_path: Path to the output file (overrides value from __init__)
        """
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
            f"[NotebookScriptWriter]: Started recording to file '{self.output_path}'"
        )

    def stop_recording(self) -> Optional[str]:
        """
        Stop recording and save accumulated code to file.

        Returns:
            Path to the created file or None on error
        """
        if not self._recording:
            logger.warning("[NotebookScriptWriter]: Recording was not started")
            return None

        # collect cells recorded since start, excluding start/end control magic cells
        try:
            df = self.cell_history.view()
        except Exception as e:
            logger.error(f"[NotebookScriptWriter]: Failed to access CellHistory: {e}")
            return None

        if df is None or df.empty:
            logger.warning("[NotebookScriptWriter]: No cells in CellHistory")
            return None

        # Select cells with index >= start and exclude cells that contain control magics
        def is_control_cell(cell_magics):
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

        selected = []
        for _, row in df.iterrows():
            try:
                idx = int(row.get("cell_index"))
            except Exception:
                continue
            if self._start_cell_index is not None and idx < self._start_cell_index:
                continue
            if is_control_cell(row.get("cell_magics")):
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
            logger.warning("[NotebookScriptWriter]: No recorded cells to save")
            # reset state
            self._recording = False
            self._start_cell_index = None
            return None

        try:
            self._write_to_file(selected)
            logger.info(
                f"[NotebookScriptWriter]: Recorded {len(selected)} cells "
                f"to file '{self.output_path}'"
            )
            return self.output_path
        except Exception as e:
            logger.error(
                f"[NotebookScriptWriter]: Error writing file: {e}"
            )
            return None
        finally:
            # reset state
            self._recording = False
            self._start_cell_index = None

    # compatibility with service.end_write_script that calls script_writer.stop()
    def stop(self) -> Optional[str]:
        return self.stop_recording()

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

                from jumper_extension.core.service import build_perfmonitor_service
                service = build_perfmonitor_service(
                    plots_disabled=True,
                    plots_disabled_reason="Plotting disabled in generated script.",
                    display_disabled=True,
                    display_disabled_reason="Display disabled in generated script."
                )

            """)
            f.write(header)

            # Write code from each recorded cell
            for cell in recorded_cells:
                f.write(f"# Cell {cell['index']}\n")
                ts = cell['timestamp']
                f.write(f"# Recorded at: {ts.strftime('%H:%M:%S') if isinstance(ts, datetime) else ts}\n")
                raw_cell = cell.get("raw_cell", "")
                cell_magics = cell.get("cell_magics") or []
                # compute should_skip_report: True if cell contains only line magics (non-empty lines start with '%')
                non_empty_lines = [ln for ln in raw_cell.splitlines() if ln.strip()]
                is_pure_magic = bool(non_empty_lines) and all(ln.lstrip().startswith("%") for ln in non_empty_lines)
                f.write(
                    "service.on_pre_run_cell("
                    f"{raw_cell!r}, "
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
                f.write("service.on_post_run_cell('')\n")
                f.write("\n")


    def _transform_cell_code(self, raw_cell: str, cell_magics: List[str]) -> str:
        """
        Replace captured magic commands with service calls while keeping
        the rest of the code intact.
        """
        if not raw_cell:
            return ""

        # Build a lookup from magic line prefix to service call string
        # We rely on CellHistory.cell_magics entries having the original magic lines (e.g. "%perfmonitor_start 1.0")
        replacements = {}
        for magic in cell_magics:
            # Normalize leading '%'
            stripped_no_pct = magic[1:] if magic.startswith("%") else magic
            parts = stripped_no_pct.split(maxsplit=1)
            cmd = parts[0]
            args = parts[1] if len(parts) > 1 else ""
            # construct a Python call to the service method
            # prefer passing the whole "line" string of arguments
            if args:
                call = f"service.{cmd}({args!r})"
            else:
                # Methods generally accept a single 'line' argument; pass empty string for uniformity
                call = f'service.{cmd}("")'
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