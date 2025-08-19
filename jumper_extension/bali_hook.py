import os
import json
import glob
from typing import List, Dict, Tuple
import matplotlib as mpl


class BaliResultsParser:
    def __init__(self, base_search_path: str = "."):
        self.base_search_path = base_search_path
        rgb = [
            mpl.colors.to_rgb(c)
            for c in ('#EADFB4', '#9BB0C1', '#F6995C', '#874C62')
        ]
        self.colormap = mpl.colors.LinearSegmentedColormap.from_list(
            'custom_cmap', rgb
        )
        
    def _find_bali_directories(self, pid: int) -> List[str]:
        pid_dir = os.path.join(self.base_search_path, "bali_results", str(pid))
        if os.path.exists(pid_dir):
            idx_dirs = [d for d in glob.glob(os.path.join(pid_dir, "*")) if os.path.isdir(d)]
            try:
                idx_dirs.sort(key=lambda x: int(os.path.basename(x)))
            except ValueError:
                idx_dirs.sort()
            return idx_dirs

        old_dirs = glob.glob(os.path.join(self.base_search_path, "bali_results_*"))
        return sorted(old_dirs, reverse=True)
    
    def parse_benchmark_results_file(self, filepath: str) -> Dict:
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError, IOError):
            return {}
    
    def extract_timing_segments(self, benchmark_data: Dict) -> List[Dict]:
        segments: List[Dict] = []
        for framework, framework_data in benchmark_data.items():
            if not isinstance(framework_data, dict):
                continue
            for iteration_key, iteration_data in framework_data.items():
                if not isinstance(iteration_data, dict):
                    continue
                start_time = iteration_data.get('start_time')
                end_time = iteration_data.get('end_time')
                tokens_per_sec = iteration_data.get('token_per_sec')
                if start_time is None or end_time is None or tokens_per_sec is None:
                    continue
                segments.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'tokens_per_sec': tokens_per_sec,
                    'framework': framework,
                    'iteration': iteration_key,
                })
        return segments
    
    def collect_all_bali_segments(self, pid: int) -> List[Dict]:
        result_dirs = self._find_bali_directories(pid)
        if not result_dirs:
            return []
        all_segments = [
            segment
            for result_dir in result_dirs
            for segment in self._collect_segments_from_directory(result_dir)
        ]
        return sorted(all_segments, key=lambda x: x['start_time'])
    
    def _collect_segments_from_directory(self, directory: str) -> List[Dict]:
        segments: List[Dict] = []
        pattern = os.path.join(directory, "*/*/*/*/benchmark_results.json")
        for result_file in glob.glob(pattern):
            benchmark_data = self.parse_benchmark_results_file(result_file)
            if benchmark_data:
                segments.extend(self.extract_timing_segments(benchmark_data))
        return segments
    
    def get_tokens_per_sec_range(self, segments: List[Dict]) -> Tuple[float, float]:
        if not segments:
            return 0.0, 100.0
        values = [seg['tokens_per_sec'] for seg in segments]
        return min(values), max(values)
    
    def get_color_for_tokens_per_sec(self, tokens_per_sec: float, vmin: float, vmax: float) -> Tuple[float, float, float, float]:
        normalized_value = 0.5 if vmax == vmin else (tokens_per_sec - vmin) / (vmax - vmin)
        normalized_value = max(0.0, min(1.0, normalized_value))
        return self.colormap(normalized_value)