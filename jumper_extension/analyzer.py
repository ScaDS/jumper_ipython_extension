from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class PerformanceTag(Enum):
    """Performance tags for classifying cells"""
    NORMAL = "normal"
    CPU_BOUND = "cpu-bound"
    MEMORY_BOUND = "memory-bound"
    GPU_UTIL_BOUND = "gpu-util-bound"
    GPU_MEMORY_BOUND = "gpu-memory-bound"
    IDLE = "idle"

    def __str__(self):
        return self.value


@dataclass
class TagScore:
    """Tag with its score for ranking"""
    tag: PerformanceTag
    score: float


class PerformanceAnalyzer:
    """Performance analyzer to determine workload type using relative thresholds"""

    MIB_TO_BYTES = 1024 * 1024

    # Default thresholds (relative to system limits)
    DEFAULT_THRESHOLDS = {
        'memory_ratio': 0.80,  # memory limit
        'cpu_ratio': 0.70,  # CPU capacity
        'gpu_util_ratio': 0.70,  # GPU utilization
        'gpu_memory_ratio': 0.80,  # GPU memory
        'idle_threshold': 0.05,  # idle detection
        'min_significant': 0.3,  # Minimum ratio for tag to be considered significant
    }

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize analyzer with relative thresholds

        Args:
            thresholds: Custom threshold values (uses defaults if None)
        """
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}

    def analyze_cell_performance(self, perfdata, memory_limit: float,
                                 gpu_memory_limit: Optional[float] = None) -> List[TagScore]:
        """
        Analyze cell performance and determine tags

        Args:
            perfdata: DataFrame with performance data
            memory_limit: System memory limit in GB
            gpu_memory_limit: GPU memory limit in GB (if available)

        Returns:
            List[TagScore]: Ranked performance tags for the cell
        """
        if perfdata.empty:
            return [TagScore(PerformanceTag.IDLE, 100.0)]

        # Compute normalized metrics
        metrics = self._compute_metrics(perfdata, gpu_memory_limit)

        # Calculate resource utilization ratios
        ratios = self._calculate_utilization_ratios(metrics, memory_limit, gpu_memory_limit)

        # Create the ranked tags list
        ranked_tags = self._create_ranked_tags(ratios)

        return ranked_tags

    @staticmethod
    def _compute_metrics(
            perfdata,
            gpu_memory_limit: Optional[float]
    ) -> Dict[str, float]:
        """Compute raw performance metrics"""
        metrics = {}

        # CPU metrics
        if 'cpu_util_avg' in perfdata.columns:
            metrics['cpu_avg'] = perfdata['cpu_util_avg'].mean()

        # Memory metrics
        if 'memory' in perfdata.columns:
            metrics['memory_avg_gb'] = perfdata['memory'].mean()

        # GPU metrics
        if 'gpu_util_avg' in perfdata.columns:
            metrics['gpu_util_avg'] = perfdata['gpu_util_avg'].mean()

        if 'gpu_mem_avg' in perfdata.columns and gpu_memory_limit:
            metrics['gpu_memory_avg_gb'] = perfdata['gpu_mem_avg'].mean()

        return metrics

    def _calculate_utilization_ratios(self, metrics: Dict[str, float],
                                      memory_limit: float,
                                      gpu_memory_limit: Optional[float]) -> Dict[str, float]:
        """Calculate utilization ratios relative to system limits"""
        ratios = {}

        # Memory ratio (current usage / limit)
        memory_avg = metrics.get('memory_avg_gb', 0)
        ratios['memory'] = self._safe_ratio(memory_avg, memory_limit)

        # CPU ratio (utilization / 100%)
        cpu_avg = metrics.get('cpu_avg', 0)
        ratios['cpu'] = self._safe_ratio(cpu_avg, 100.0)

        # GPU utilization ratio
        gpu_util = metrics.get('gpu_util_avg', 0)
        ratios['gpu_util'] = self._safe_ratio(gpu_util, 100.0)

        # GPU memory ratio
        if gpu_memory_limit and gpu_memory_limit > 0:
            gpu_memory = metrics.get('gpu_memory_avg_gb', 0)
            ratios['gpu_memory'] = self._safe_ratio(gpu_memory, gpu_memory_limit)
        else:
            ratios['gpu_memory'] = 0.0

        return ratios

    @staticmethod
    def _safe_ratio(measured: float, maximum: float) -> float:
        """Safely calculate ratio with error handling"""
        try:
            if maximum is None or maximum <= 0 or measured is None:
                return 0.0
            return min(1.0, max(0.0, measured / maximum))
        except (TypeError, ZeroDivisionError):
            return 0.0

    def _create_ranked_tags(self, ratios: Dict[str, float]) -> List[TagScore]:
        """Create the ranked list of tags based on ratios (0.0-1.0 scale)"""

        # Check for idle (all ratios below the threshold)
        idle_threshold = self.thresholds['idle_threshold']
        if all(ratio < idle_threshold for ratio in ratios.values()):
            return [TagScore(PerformanceTag.IDLE, idle_threshold)]

        # Sort by descending ratios
        sorted_ratios = sorted(ratios.items(), key=lambda x: x[1], reverse=True)

        # Check if any resource exceeds its threshold (bound detection)
        threshold_exceeded = False
        for resource, ratio in sorted_ratios:
            threshold_key = f'{resource}_ratio'
            if threshold_key in self.thresholds:
                threshold = self.thresholds[threshold_key]
                if ratio >= threshold:
                    threshold_exceeded = True
                    break

        # If no threshold is exceeded, classify as NORMAL
        if not threshold_exceeded:
            return [TagScore(PerformanceTag.NORMAL, sorted_ratios[0][1] if sorted_ratios else 0.0)]

        # Create ranked tags for resources that exceed minimum threshold
        tag_mapping = {
            'cpu': PerformanceTag.CPU_BOUND,
            'memory': PerformanceTag.MEMORY_BOUND,
            'gpu_util': PerformanceTag.GPU_UTIL_BOUND,
            'gpu_memory': PerformanceTag.GPU_MEMORY_BOUND,
        }

        ranked_tags = []


        for resource, ratio in sorted_ratios:
            if ratio >= self.thresholds['min_significant']:
                tag = tag_mapping.get(resource)
                if tag:
                    ranked_tags.append(TagScore(tag, ratio))

        # If no tags above the threshold, fallback to NORMAL
        if not ranked_tags:
            ranked_tags = [TagScore(PerformanceTag.NORMAL, sorted_ratios[0][1] if sorted_ratios else 0.0)]

        return ranked_tags
