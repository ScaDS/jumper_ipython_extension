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
            [
                mpl.colors.to_rgb(c)
                for c in ("#EADFB4", "#9BB0C1", "#F6995C", "#874C62")
            ],
        )

    def _find_bali_directories(self, pid: int) -> List[str]:
        pid_dir = os.path.join(self.base_search_path, "bali_results", str(pid))
        if os.path.exists(pid_dir):
            idx_dirs = [
                d
                for d in glob.glob(os.path.join(pid_dir, "*"))
                if os.path.isdir(d)
            ]
            if all(os.path.basename(d).isdigit() for d in idx_dirs):
                idx_dirs.sort(key=lambda x: int(os.path.basename(x)))
            else:
                idx_dirs.sort()
            return idx_dirs
        return sorted(
            glob.glob(os.path.join(self.base_search_path, "bali_results_*")),
            reverse=True,
        )

    def parse_benchmark_results_file(self, filepath: str) -> Dict:
        try:
            with open(filepath, "r") as f:
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
                s = iteration_data.get("start_time")
                e = iteration_data.get("end_time")
                t = iteration_data.get("token_per_sec")
                if s is None or e is None or t is None:
                    continue

                # Try to extract optional metadata if present in various common schemas
                def _first_present(d: Dict, keys):
                    for k in keys:
                        if isinstance(d, dict) and k in d:
                            return d.get(k)
                    return None

                # Candidates where metadata might live
                meta_candidates = [
                    iteration_data,
                    iteration_data.get("config", {}),
                    iteration_data.get("params", {}),
                    framework_data,
                ]

                meta: Dict = {"model": None, "batch_size": None, "input_len": None, "output_len": None}
                for cand in meta_candidates:
                    if meta["model"] is None:
                        meta["model"] = _first_present(
                            cand,
                            [
                                "model",
                                "model_name",
                                "model_id",
                                "llm",
                                "hf_model",
                            ],
                        )
                    if meta["batch_size"] is None:
                        meta["batch_size"] = _first_present(
                            cand, ["batch_size", "batch", "bs", "micro_batch_size"]
                        )
                    if meta["input_len"] is None:
                        meta["input_len"] = _first_present(
                            cand,
                            [
                                "input_len",
                                "input_length",
                                "prompt_tokens",
                                "input_tokens",
                                "num_input_tokens",
                            ],
                        )
                    if meta["output_len"] is None:
                        meta["output_len"] = _first_present(
                            cand,
                            [
                                "output_len",
                                "output_length",
                                "generated_tokens",
                                "output_tokens",
                                "new_tokens",
                                "num_output_tokens",
                            ],
                        )

                def _coerce_int(value):
                    try:
                        return int(value)
                    except Exception:
                        return value

                if meta["batch_size"] is not None:
                    meta["batch_size"] = _coerce_int(meta["batch_size"])
                if meta["input_len"] is not None:
                    meta["input_len"] = _coerce_int(meta["input_len"])
                if meta["output_len"] is not None:
                    meta["output_len"] = _coerce_int(meta["output_len"])

                seg = {
                    "start_time": s,
                    "end_time": e,
                    "duration": e - s,
                    "tokens_per_sec": t,
                    "framework": framework,
                    "iteration": iteration_key,
                    # Optional metadata for hover tooltips
                    "model": meta["model"],
                    "batch_size": meta["batch_size"],
                    "input_len": meta["input_len"],
                    "output_len": meta["output_len"],
                }
                segments.append(seg)
        return segments

    def collect_all_bali_segments(self, pid: int) -> List[Dict]:
        result_dirs = self._find_bali_directories(pid)
        if not result_dirs:
            return []
        segments = [
            seg
            for d in result_dirs
            for seg in self._collect_segments_from_directory(d)
        ]
        return sorted(segments, key=lambda x: x["start_time"])

    def _collect_segments_from_directory(self, directory: str) -> List[Dict]:
        segments: List[Dict] = []
        for result_file in glob.glob(
            os.path.join(directory, "*/*/*/*/benchmark_results.json")
        ):
            data = self.parse_benchmark_results_file(result_file)
            if data:
                segments.extend(self.extract_timing_segments(data))
        return segments

    def get_tokens_per_sec_range(
        self, segments: List[Dict]
    ) -> Tuple[float, float]:
        if not segments:
            return 0.0, 100.0
        values = [seg["tokens_per_sec"] for seg in segments]
        return min(values), max(values)

    def get_color_for_tokens_per_sec(
        self, tokens_per_sec: float, vmin: float, vmax: float
    ) -> Tuple[float, float, float, float]:
        normalized_value = (
            0.5
            if vmax == vmin
            else max(0.0, min(1.0, (tokens_per_sec - vmin) / (vmax - vmin)))
        )
        return self.colormap(normalized_value)
