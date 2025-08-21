import pandas as pd
from typing import List, Dict, Tuple, Any
from itables import show

from .bali_hook import BaliResultsParser


class BaliAdapter:
    """
    Adapter class that provides a clean interface for BALI functionality.
    """

    def __init__(self):
        self.parser = BaliResultsParser()
        self._segments_df = pd.DataFrame(
            columns=[
                "model",
                "framework",
                "batch_size",
                "input_len",
                "output_len",
                "iteration",
                "start_time",
                "end_time",
                "duration",
                "tokens_per_sec",
            ]
        )

    def get_segments_dataframe(self) -> pd.DataFrame:
        """Get the current BALI segments as a DataFrame."""
        return self._segments_df.copy()

    def refresh_segments_from_disk(self, pid: int) -> int:
        """
        Refresh BALI segments from disk for the given process ID.
        """
        segments = self.parser.collect_all_bali_segments(pid)

        # Build DataFrame directly from segments and align to canonical column order
        df = pd.DataFrame(segments)
        self._segments_df = df.reindex(columns=self._segments_df.columns)

        return len(self._segments_df)

    def get_segments_for_visualization(self, pid: int) -> List[Dict]:
        """
        Get BALI segments in the format needed for visualization.
        """
        if self._segments_df.empty:
            self.refresh_segments_from_disk(pid)

        df = self._segments_df
        df = df[df["start_time"].notna() & df["end_time"].notna()]
        return df.to_dict(orient="records")

    def get_tokens_per_sec_range(
        self, segments: List[Dict]
    ) -> Tuple[float, float]:
        """Get the min/max tokens per second range for coloring."""
        return self.parser.get_tokens_per_sec_range(segments)

    def get_color_for_tokens_per_sec(
        self, tokens_per_sec: float, vmin: float, vmax: float
    ) -> Tuple[float, float, float, float]:
        """Get color for a given tokens per second value."""
        return self.parser.get_color_for_tokens_per_sec(
            tokens_per_sec, vmin, vmax
        )

    def get_colormap(self):
        """Get the colormap used for visualization."""
        return self.parser.colormap

    def compress_segments(
        self,
        segments: List[Dict],
        cell_range: Tuple[int, int],
        perfdata: Any,
        cell_history: Any,
        current_time_offset: float = 0,
    ) -> List[Dict]:
        """
        Compress BALI segments to match compressed time axis.
        """
        if not segments:
            return []

        start_idx, end_idx = cell_range
        cell_data = cell_history.view(start_idx, end_idx + 1)
        compressed, current_time, processed = [], current_time_offset, set()

        for _, cell in cell_data.iterrows():
            if perfdata[
                (perfdata["time"] >= cell["start_time"])
                & (perfdata["time"] <= cell["end_time"])
            ].empty:
                continue

            cell_start, cell_end = cell["start_time"], cell["end_time"]
            cell_duration = cell_end - cell_start

            for i, seg in enumerate(segments):
                seg_id = (i, cell["index"])
                if seg_id in processed:
                    continue

                seg_start, seg_end = seg["start_time"], seg["end_time"]
                overlap_start = max(seg_start, cell_start)
                overlap_end = min(seg_end, cell_end)

                if overlap_start < overlap_end:
                    start = current_time + (overlap_start - cell_start)
                    duration = overlap_end - overlap_start
                    compressed.append(
                        {
                            "start_time": start,
                            "end_time": start + duration,
                            "duration": duration,
                            "tokens_per_sec": seg["tokens_per_sec"],
                            "framework": seg["framework"],
                            "iteration": seg["iteration"],
                            "model": seg.get("model"),
                            "batch_size": seg.get("batch_size"),
                            "input_len": seg.get("input_len"),
                            "output_len": seg.get("output_len"),
                        }
                    )
                    processed.add(seg_id)
            current_time += cell_duration
        return compressed


class BaliVisualizationMixin:
    """
    Mixin class that provides BALI visualization capabilities.

    This mixin can be added to visualization classes to provide BALI
    functionality without directly coupling the core visualization code.
    """

    def __init__(self, *args, bali_adapter=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bali_adapter = bali_adapter or BaliAdapter()
        self._compressed_bali_segments = []

    def _load_bali_segments(self) -> List[Dict]:
        """Load BALI segments from disk."""
        return self.bali_adapter.get_segments_for_visualization(self.monitor.pid)


class BaliMagicsMixin:
    """
    Mixin class that provides BALI magic commands.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bali_adapter = BaliAdapter()

    def _bali_refresh_from_disk(self):
        """Collect BALI segments from disk."""
        pid = getattr(self.monitor, "pid", 0) if self.monitor else 0
        return self.bali_adapter.refresh_segments_from_disk(pid)

    def _bali_segments(self, line: str):
        """Handle the bali_segments magic command."""
        bali_segments = self.bali_adapter.get_segments_dataframe()
        if bali_segments.empty:
            self._bali_refresh_from_disk()
            bali_segments = self.bali_adapter.get_segments_dataframe()

        if bali_segments.empty:
            return print("No BALI segments to display.")

        show(
            bali_segments,
            layout={"topStart": "search", "topEnd": None},
        )


    def _bali_run(self, line: str):
        """Handle the bali_run magic command."""
        count = self._bali_refresh_from_disk()
        print(f"BALI segments: {count} rows")
