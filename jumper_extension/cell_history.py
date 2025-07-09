import json
import time
import os
import warnings

import pandas as pd
from itables import show


class CellHistory:
    def __init__(self):
        self.data = pd.DataFrame(
            columns=["index", "raw_cell", "start_time", "end_time", "duration"]
        )
        self.current_cell = None

    def start_cell(self, raw_cell):
        self.current_cell = {
            "index": len(self.data),
            "raw_cell": raw_cell,
            "start_time": time.time(),
            "end_time": None,
            "duration": None,
        }

    def end_cell(self, result):
        if self.current_cell:
            self.current_cell["end_time"] = time.time()
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
        for i, (_, cell) in enumerate(self.data.iterrows()):
            print(f"Cell #{int(cell['index'])} - Duration: {cell['duration']:.2f}s")
            print("-" * 40)
            print(cell["raw_cell"])
            print("=" * 40)

    def show_itable(self):
        if self.data.empty:
            print("No cell history to display.")
            return

        data = []
        for _, row in self.data.iterrows():
            duration = row["end_time"] - row["start_time"]
            data.append({
                "Cell index": row["index"],
                "Duration (s)": f"{duration:.2f}",
                "Start Time": time.strftime('%H:%M:%S', time.localtime(row["start_time"])),
                "End Time": time.strftime('%H:%M:%S', time.localtime(row["end_time"])),
                "Code": row["raw_cell"].replace("\n", "<br>")
            })

        df = pd.DataFrame(data)
        show(
            df,
            layout={"topStart": "search", "topEnd": None},
            columnDefs=[
                {"targets": [4], "className": "dt-left"}  # 4 - "Code" index
            ],
            escape=False,
        )

    def export(self, filename="cell_history.json"):
        if self.data.empty:
            print(f"No cell history to export to {filename}")
            return

        # Determine format from filename extension
        _, ext = os.path.splitext(filename)
        format = ext.lower().lstrip(".")

        if format == "json":
            with open(filename, "w") as f:
                json.dump(self.data.to_dict("records"), f, indent=2)
        elif format == "csv":
            self.data.to_csv(filename, index=False)
        else:
            print(f"Unsupported format: {format}. Use 'json' or 'csv'")
            return

        print(f"Cell history exported to {filename}")

    def __len__(self):
        return len(self.data)
