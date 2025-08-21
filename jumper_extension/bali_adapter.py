"""
BALI Integration Adapter

This module provides a clean interface for BALI functionality, isolating all
BALI-specific code to minimize merge conflicts when rebasing on main.
"""

import pandas as pd
from typing import List, Dict, Optional, Tuple, Any

try:
    from .bali_hook import BaliResultsParser
    BALI_AVAILABLE = True
except ImportError:
    BALI_AVAILABLE = False
    BaliResultsParser = None


class BaliAdapter:
    """
    Adapter class that provides a clean interface for BALI functionality.
    
    This class encapsulates all BALI-related operations and provides a stable
    interface that can be easily maintained separately from the core codebase.
    """
    
    def __init__(self):
        if BALI_AVAILABLE:
            self.parser = BaliResultsParser()
        else:
            self.parser = None
        self._segments_df = pd.DataFrame(columns=[
            "model", "framework", "batch size", "input", "output", 
            "iteration", "start time", "end time", "duration", "tok/s"
        ])
    
    def is_available(self) -> bool:
        """Check if BALI functionality is available."""
        return BALI_AVAILABLE and self.parser is not None
    
    def get_segments_dataframe(self) -> pd.DataFrame:
        """Get the current BALI segments as a DataFrame."""
        return self._segments_df.copy()
    
    def refresh_segments_from_disk(self, pid: int) -> int:
        """
        Refresh BALI segments from disk for the given process ID.
        
        Args:
            pid: Process ID to search for BALI results
            
        Returns:
            Number of segments loaded
        """
        if not self.is_available():
            return 0
        segments = self.parser.collect_all_bali_segments(pid)
        
        self._segments_df = pd.DataFrame([
            {
                "model": seg.get("model"),
                "framework": seg.get("framework"),
                "batch size": seg.get("batch_size"),
                "input": seg.get("input_len"),
                "output": seg.get("output_len"),
                "iteration": seg.get("iteration"),
                "start time": seg.get("start_time"),
                "end time": seg.get("end_time"),
                "duration": seg.get("duration"),
                "tok/s": seg.get("tokens_per_sec"),
            }
            for seg in segments
        ], columns=self._segments_df.columns)
        
        return len(self._segments_df)
    
    def get_segments_for_visualization(self, pid: int) -> List[Dict]:
        """
        Get BALI segments in the format needed for visualization.
        
        Args:
            pid: Process ID to search for BALI results
            
        Returns:
            List of segment dictionaries for visualization
        """
        if not self.is_available():
            return []
        if not self._segments_df.empty:
            return [
                {
                    "start_time": r.get("start time"),
                    "end_time": r.get("end time"),
                    "duration": r.get("duration"),
                    "tokens_per_sec": r.get("tok/s"),
                    "framework": r.get("framework"),
                    "iteration": r.get("iteration"),
                    "model": r.get("model"),
                    "batch_size": r.get("batch size"),
                    "input_len": r.get("input"),
                    "output_len": r.get("output"),
                }
                for _, r in self._segments_df.iterrows()
                if r.get("start time") is not None and r.get("end time") is not None
            ]
        return self.parser.collect_all_bali_segments(pid)
    
    def get_tokens_per_sec_range(self, segments: List[Dict]) -> Tuple[float, float]:
        """Get the min/max tokens per second range for coloring."""
        if not self.is_available():
            return (0.0, 100.0)
        return self.parser.get_tokens_per_sec_range(segments)
    
    def get_color_for_tokens_per_sec(self, tokens_per_sec: float, vmin: float, vmax: float) -> Tuple[float, float, float, float]:
        """Get color for a given tokens per second value."""
        if not self.is_available():
            return (0.5, 0.5, 0.5, 1.0)  # Gray color as fallback
        return self.parser.get_color_for_tokens_per_sec(tokens_per_sec, vmin, vmax)
    
    def get_colormap(self):
        """Get the colormap used for visualization."""
        if not self.is_available():
            # Return a simple fallback colormap
            import matplotlib.pyplot as plt
            return plt.cm.viridis
        return self.parser.colormap
    
    def compress_segments(self, segments: List[Dict], cell_range: Tuple[int, int], 
                         cell_data: Any, perfdata: Any, current_time_offset: float = 0) -> List[Dict]:
        """
        Compress BALI segments to match compressed time axis.
        
        Args:
            segments: List of BALI segments
            cell_range: Tuple of (start_idx, end_idx)
            cell_data: Cell history data
            perfdata: Performance data
            current_time_offset: Current time offset for compression
            
        Returns:
            List of compressed segments
        """
        if not segments:
            return []
        
        start_idx, end_idx = cell_range
        compressed, current_time, processed = [], current_time_offset, set()
        
        for _, cell in cell_data.iterrows():
            if perfdata[(perfdata["time"] >= cell["start_time"]) & 
                       (perfdata["time"] <= cell["end_time"])].empty:
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
                    compressed.append({
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
                    })
                    processed.add(seg_id)
            current_time += cell_duration
        return compressed


class BaliVisualizationMixin:
    """
    Mixin class that provides BALI visualization capabilities.
    
    This mixin can be added to visualization classes to provide BALI
    functionality without directly coupling the core visualization code.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bali_adapter = BaliAdapter() if self._should_enable_bali() else None
        self._compressed_bali_segments = []
    
    def _should_enable_bali(self) -> bool:
        """Determine if BALI should be enabled."""
        try:
            adapter = BaliAdapter()
            return adapter.is_available()
        except Exception:
            return False
    
    def _load_bali_segments(self) -> List[Dict]:
        """Load BALI segments from disk."""
        if not self.bali_adapter:
            return []
        
        return self.bali_adapter.get_segments_for_visualization(getattr(self.monitor, 'pid', 0))
    
    def _compress_bali_segments(self, bali_segments: List[Dict], cell_range: Tuple[int, int], perfdata: Any) -> List[Dict]:
        """Compress BALI segments to match compressed time axis."""
        if not self.bali_adapter or not bali_segments:
            return []
        
        # This requires cell_data which needs to be passed from the calling context
        # For now, return the original implementation logic
        start_idx, end_idx = cell_range
        cell_data = self.cell_history.view(start_idx, end_idx + 1)
        return self.bali_adapter.compress_segments(bali_segments, cell_range, cell_data, perfdata)


class BaliMagicsMixin:
    """
    Mixin class that provides BALI magic commands.
    
    This mixin can be added to magic classes to provide BALI
    functionality without directly coupling the core magic code.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bali_adapter = BaliAdapter() if self._should_enable_bali() else None
        self.bali_segments = pd.DataFrame(columns=[
            "model", "framework", "batch size", "input", "output", 
            "iteration", "start time", "end time", "duration", "tok/s"
        ])
    
    def _should_enable_bali(self) -> bool:
        """Determine if BALI should be enabled."""
        try:
            adapter = BaliAdapter()
            return adapter.is_available()
        except Exception:
            return False
    
    def _bali_refresh_from_disk(self):
        """Collect BALI segments from disk and persist to self.bali_segments."""
        if not self.bali_adapter:
            return
        
        pid = getattr(self.monitor, "pid", 0) if self.monitor else 0
        count = self.bali_adapter.refresh_segments_from_disk(pid)
        self.bali_segments = self.bali_adapter.get_segments_dataframe()
        return count
    
    def _handle_bali_segments_command(self, line: str):
        """Handle the bali_segments magic command."""
        if not self.bali_adapter:
            print("BALI functionality is not available.")
            return
        
        # Populate from disk if empty
        if self.bali_segments.empty:
            self._bali_refresh_from_disk()
        
        if self.bali_segments.empty:
            return print("No BALI segments to display.")
        
        # Import here to avoid dependency issues if itables is not available
        try:
            from itables import show
            show(self.bali_segments, layout={"topStart": "search", "topEnd": None})
        except ImportError:
            print("itables not available, displaying as regular DataFrame:")
            print(self.bali_segments)
    
    def _handle_bali_run_command(self, line: str):
        """Handle the bali_run magic command."""
        if not self.bali_adapter:
            print("BALI functionality is not available.")
            return
        
        count = self._bali_refresh_from_disk()
        print(f"BALI segments: {count} rows")