import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime

logger = logging.getLogger("extension")


class NotebookScriptWriter:
    """
    Class for writing notebook content to a Python script.
    
    Collects code from cells and saves it to a Python file with optional
    metadata about execution time and cell numbers.
    """
    
    def __init__(self, output_path: Optional[str] = None):
        """
        Initialize the writer.
        
        Args:
            output_path: Path to the output Python file.
                        If None, will be generated automatically.
        """
        self.output_path = output_path
        self.recorded_cells: List[dict] = []
        
    def start_recording(self, output_path: Optional[str] = None):
        """
        Start recording code from cells.
        
        Args:
            output_path: Path to the output file (overrides value from __init__)
        """
        if output_path:
            self.output_path = output_path
            
        if not self.output_path:
            # Generate default filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = f"notebook_script_{timestamp}.py"
            
        self.recorded_cells = []

        logger.info(
            f"[NotebookScriptWriter]: Started recording to file '{self.output_path}'"
        )
        
    def stop_recording(self) -> Optional[str]:
        """
        Stop recording and save accumulated code to file.
        
        Returns:
            Path to the created file or None on error
        """
        if not self.recorded_cells:
            logger.warning(
                "[NotebookScriptWriter]: No recorded cells to save"
            )
            return None
            
        try:
            self._write_to_file()
            logger.info(
                f"[NotebookScriptWriter]: Recorded {self.cell_count} cells "
                f"to file '{self.output_path}'"
            )
            return self.output_path
        except Exception as e:
            logger.error(
                f"[NotebookScriptWriter]: Error writing file: {e}"
            )
            return None

    def _parse_magics(self, code: str):
        pass
        
    def _write_to_file(self):
        """
        Write accumulated cells to Python file.
        """
        output_path = Path(self.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            # File header
            f.write("#!/usr/bin/env python3\n")
            f.write('"""')
            f.write("Auto-generated script from Jupyter notebook\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if self.start_time:
                f.write(
                    f"Recording started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
            f.write(f"Total cells: {self.cell_count}\n")
            f.write('"""\n\n')
            
            # Write code from each cell
            for cell in self.recorded_cells:
                f.write(f"# Cell {cell['index']}\n")
                f.write(f"# Recorded at: {cell['timestamp'].strftime('%H:%M:%S')}\n")
                f.write(f"{cell['code']}\n")
                f.write("\n")
