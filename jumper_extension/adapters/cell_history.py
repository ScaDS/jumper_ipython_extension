import json
import logging
import logging.config
import os
import time
import warnings
from typing import List, Optional

import pandas as pd
from itables import show, options

from jumper_extension.core.messages import (
    ExtensionErrorCode,
    ExtensionInfoCode,
    EXTENSION_ERROR_MESSAGES,
    EXTENSION_INFO_MESSAGES,
)
from jumper_extension.utilities import load_dataframe_from_file

logger = logging.getLogger("extension")


class CellHistory:
    def __init__(self):
        self._columns = [
            "cell_index",
            "raw_cell",
            "start_time",
            "end_time",
            "duration",
        ]
        self.data = pd.DataFrame(columns=self._columns)
        self.file_readers = {
            "json": pd.read_json,
            "csv": pd.read_csv,
        }
        self.current_cell = None

    def start_cell(self, raw_cell: str, cell_magics: List[str]):
        self.current_cell = {
            "cell_index": len(self.data),
            "cell_magics": cell_magics,
            "raw_cell": raw_cell,
            "start_time": time.perf_counter(),
            "end_time": None,
            "duration": None,
        }

    def end_cell(self, result):
        if self.current_cell:
            self.current_cell["end_time"] = time.perf_counter()
            self.current_cell["duration"] = (
                self.current_cell["end_time"] - self.current_cell["start_time"]
            )

            new_row = pd.DataFrame([self.current_cell])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
                self.data = pd.concat([self.data, new_row], ignore_index=True)
            self.current_cell = None

    def view(self, start=None, end=None):
        if start is None and end is None:
            return self.data
        return self.data.iloc[start:end]

    def print(self):
        for _, cell in self.data.iterrows():
            print(
                f"Cell #{int(cell['cell_index'])} - Duration: "
                f"{cell['duration']:.2f}s"
            )
            print("-" * 40)
            print(cell["raw_cell"])
            print("=" * 40)

    def show_itable(self):
        if self.data.empty:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_CELL_HISTORY]
            )
            return

        data = []
        for _, row in self.data.iterrows():
            duration = row["end_time"] - row["start_time"]
            data.append(
                {
                    "Cell index": row["cell_index"],
                    "Duration (s)": f"{duration:.2f}",
                    "Start Time": time.strftime(
                        "%H:%M:%S", time.localtime(row["start_time"])
                    ),
                    "End Time": time.strftime(
                        "%H:%M:%S", time.localtime(row["end_time"])
                    ),
                    "Code": row["raw_cell"].replace("\n", "<br>"),
                }
            )

        df = pd.DataFrame(data)

        # To avoid warnings about a non-documented 'escape' option in a notebook
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SyntaxWarning, module="itables\\.typing")
            show(
                df,
                layout={"topStart": "search", "topEnd": None},
                columnDefs=[
                    {"targets": [4], "className": "dt-left"}
                ],  # 4 - "Code" index
                escape=False,
            )

    def export(self, filename="cell_history.json"):
        if self.data.empty:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_CELL_HISTORY]
            )
            return

        # Determine format from filename extension
        _, ext = os.path.splitext(filename)
        format = ext.lower().lstrip(".")
        
        # Default to csv if no extension provided
        if not format:
            format = "csv"
            filename += ".csv"

        if format == "json":
            with open(filename, "w") as f:
                json.dump(self.data.to_dict("records"), f, indent=2)
        elif format == "csv":
            self.data.to_csv(filename, index=False)
        else:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.UNSUPPORTED_FORMAT
                ].format(
                    format=format,
                    supported_formats=", ".join(["json", "csv"]),
                )
            )
            return

        logger.info(
            EXTENSION_INFO_MESSAGES[ExtensionInfoCode.EXPORT_SUCCESS].format(
                filename=filename
            )
        )

    def load(self, filename: str) -> Optional[pd.DataFrame]:
        """Load cell history from CSV or JSON file.

        Returns:
            DataFrame if successful, None otherwise
        """
        return load_dataframe_from_file(
            filename,
            self.file_readers,
            self._columns,
            entity_name="cell history",
        )

    def __len__(self):
        return len(self.data)
