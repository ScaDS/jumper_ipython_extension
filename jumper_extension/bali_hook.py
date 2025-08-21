import os
import json
import glob
from typing import List, Dict, Tuple
import matplotlib as mpl


class BaliResultsParser:
    def __init__(self, base_search_path: str = "."):
        self.base_search_path = base_search_path
        self.colormap = mpl.colors.LinearSegmentedColormap.from_list(
            "custom_cmap",
            [mpl.colors.to_rgb(c) for c in ("#EADFB4", "#9BB0C1", "#F6995C", "#874C62")],
        )

    def _find_bali_directories(self, pid: int) -> List[str]:
        pid_dir = os.path.join(self.base_search_path, "bali_results", str(pid))
        idx_dirs = [d for d in glob.glob(os.path.join(pid_dir, "*")) if os.path.isdir(d)]
        return sorted(idx_dirs, key=lambda x: int(os.path.basename(x)) if os.path.basename(x).isdigit() else 0)

    def _load_json(self, filepath: str) -> Dict:
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except:
            return {}

    def extract_segment(self, benchmark_data: Dict, config_data: Dict) -> List[Dict]:
        segments = []
        if benchmark_data:
            # Get the single framework (first and only key)
            framework = next(iter(benchmark_data))
            framework_data = benchmark_data[framework]
            
            for iteration_key, iteration_data in framework_data.items():
                start_time = iteration_data.get("start_time")
                end_time = iteration_data.get("end_time")
                
                segments.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": (end_time - start_time) if (start_time and end_time) else None,
                    "tokens_per_sec": iteration_data.get("token_per_sec"),
                    "framework": framework,
                    "iteration": iteration_key,
                    "model": config_data.get("model_name"),
                    "batch_size": config_data.get("batch_size"),
                    "input_len": config_data.get("input_len"),
                    "output_len": config_data.get("output_len"),
                })
        else:
            # Create segment with missing timing data when benchmark_data doesn't exist
            segments.append({
                "start_time": None,
                "end_time": None,
                "duration": None,
                "tokens_per_sec": None,
                "framework": "unknown",
                "iteration": "0",
                "model": config_data.get("model_name"),
                "batch_size": config_data.get("batch_size"),
                "input_len": config_data.get("input_len"),
                "output_len": config_data.get("output_len"),
            })
        return segments

    def collect_all_bali_segments(self, pid: int) -> List[Dict]:
        result_dirs = self._find_bali_directories(pid)
        if not result_dirs:
            return []
        
        segments = []
        for directory in result_dirs:
            for config_path in glob.glob(os.path.join(directory, "*/*/*/*/config.json")):
                config_data = self._load_json(config_path)
                benchmark_path = os.path.join(os.path.dirname(config_path), "benchmark_results.json")
                benchmark_data = self._load_json(benchmark_path)
                
                segments.extend(self.extract_segment(benchmark_data, config_data))
        
        return sorted([s for s in segments if s["start_time"]], key=lambda x: x["start_time"])

    def get_tokens_per_sec_range(self, segments: List[Dict]) -> Tuple[float, float]:
        values = [s["tokens_per_sec"] for s in segments if s["tokens_per_sec"]]
        return (min(values), max(values)) if values else (0.0, 100.0)

    def get_color_for_tokens_per_sec(self, tokens_per_sec: float, vmin: float, vmax: float) -> Tuple[float, float, float, float]:
        if vmax == vmin:
            return self.colormap(0.5)
        normalized = max(0.0, min(1.0, (tokens_per_sec - vmin) / (vmax - vmin)))
        return self.colormap(normalized)
