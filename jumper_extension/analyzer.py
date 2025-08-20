import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class PerformanceTag(Enum):
    """Performance tags for classifying cells"""
    CPU_BOUND = "cpu-bound"
    MEMORY_BOUND = "memory-bound"
    GPU_BOUND = "gpu-bound"
    IO_BOUND = "io-bound"
    BALANCED = "balanced"
    IDLE = "idle"


@dataclass
class PerformanceProfile:
    """Performance profile for a cell"""
    primary_tag: PerformanceTag
    secondary_tags: List[PerformanceTag]
    confidence: float
    metrics_summary: Dict[str, float]
    bottleneck_score: Dict[str, float]


class PerformanceAnalyzer:
    """Performance analyzer to determine workload type"""

    def __init__(self,
                 cpu_threshold_high=70.0,
                 cpu_threshold_low=20.0,
                 memory_threshold_high=80.0,
                 memory_threshold_low=30.0,
                 gpu_threshold_high=70.0,
                 gpu_threshold_low=20.0,
                 io_threshold_high=50.0,
                 idle_threshold=5.0):
        """
        Initialize analyzer with threshold values

        Args:
            cpu_threshold_high: High CPU utilization threshold (%)
            cpu_threshold_low: Low CPU utilization threshold (%)
            memory_threshold_high: High memory usage threshold (%)
            memory_threshold_low: Low memory usage threshold (%)
            gpu_threshold_high: High GPU utilization threshold (%)
            gpu_threshold_low: Low GPU utilization threshold (%)
            io_threshold_high: High threshold for I/O operations
            idle_threshold: Threshold to determine system idle
        """
        self.thresholds = {
            'cpu_high': cpu_threshold_high,
            'cpu_low': cpu_threshold_low,
            'memory_high': memory_threshold_high,
            'memory_low': memory_threshold_low,
            'gpu_high': gpu_threshold_high,
            'gpu_low': gpu_threshold_low,
            'io_high': io_threshold_high,
            'idle': idle_threshold
        }

    def analyze_cell_performance(self, perfdata, memory_limit: float,
                                 gpu_memory_limit: float = None) -> PerformanceProfile:
        """
        Analyze cell performance and determine tags

        Args:
            perfdata: DataFrame with performance data
            memory_limit: Memory limit in GB
            gpu_memory_limit: GPU memory limit in GB (if available)

        Returns:
            PerformanceProfile: Performance profile of the cell
        """
        if perfdata.empty:
            return PerformanceProfile(
                primary_tag=PerformanceTag.IDLE,
                secondary_tags=[],
                confidence=1.0,
                metrics_summary={},
                bottleneck_score={}
            )

        # Compute metrics
        metrics = self._compute_metrics(perfdata, memory_limit, gpu_memory_limit)

        # Determine bottleneck scores
        bottleneck_scores = self._calculate_bottleneck_scores(metrics)

        # Classify performance
        primary_tag, secondary_tags, confidence = self._classify_performance(
            metrics, bottleneck_scores
        )

        return PerformanceProfile(
            primary_tag=primary_tag,
            secondary_tags=secondary_tags,
            confidence=confidence,
            metrics_summary=metrics,
            bottleneck_score=bottleneck_scores
        )

    def _compute_metrics(self, perfdata, memory_limit: float,
                         gpu_memory_limit: Optional[float]) -> Dict[str, float]:
        """Compute key performance metrics"""
        metrics = {}

        # CPU metrics
        if 'cpu_util_avg' in perfdata.columns:
            metrics['cpu_avg'] = perfdata['cpu_util_avg'].mean()
            metrics['cpu_max'] = perfdata['cpu_util_avg'].max()
            metrics['cpu_variance'] = perfdata['cpu_util_avg'].var()

        # Memory metrics
        if 'memory' in perfdata.columns:
            memory_usage_pct = (perfdata['memory'] / memory_limit * 100)
            metrics['memory_avg_pct'] = memory_usage_pct.mean()
            metrics['memory_max_pct'] = memory_usage_pct.max()
            metrics['memory_growth_rate'] = self._calculate_growth_rate(perfdata['memory'])

        # GPU metrics (if available)
        if 'gpu_util_avg' in perfdata.columns:
            metrics['gpu_util_avg'] = perfdata['gpu_util_avg'].mean()
            metrics['gpu_util_max'] = perfdata['gpu_util_avg'].max()

        if 'gpu_mem_avg' in perfdata.columns and gpu_memory_limit:
            gpu_mem_pct = (perfdata['gpu_mem_avg'] / gpu_memory_limit * 100)
            metrics['gpu_memory_avg_pct'] = gpu_mem_pct.mean()
            metrics['gpu_memory_max_pct'] = gpu_mem_pct.max()

        # I/O metrics
        if 'io_read' in perfdata.columns and 'io_write' in perfdata.columns:
            metrics['io_read_avg'] = perfdata['io_read'].mean()
            metrics['io_write_avg'] = perfdata['io_write'].mean()
            metrics['io_total_avg'] = metrics['io_read_avg'] + metrics['io_write_avg']

        return metrics

    @staticmethod
    def _calculate_growth_rate(series) -> float:
        """Calculate growth rate for a time series"""
        if len(series) < 2:
            return 0.0

        # Linear regression to determine trend
        x = np.arange(len(series))
        coefficients = np.polyfit(x, series, 1)
        return coefficients[0]  # Slope coefficient

    def _calculate_bottleneck_scores(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate bottleneck scores"""
        scores = {}

        # CPU score
        cpu_avg = metrics.get('cpu_avg', 0)
        if cpu_avg >= self.thresholds['cpu_high']:
            scores['cpu'] = min(100.0, cpu_avg * 1.2)
        elif cpu_avg <= self.thresholds['cpu_low']:
            scores['cpu'] = max(0.0, cpu_avg * 0.8)
        else:
            scores['cpu'] = cpu_avg

        # Memory score
        memory_pct = metrics.get('memory_avg_pct', 0)
        memory_growth = metrics.get('memory_growth_rate', 0)
        memory_score = memory_pct

        # Consider memory usage growth
        if memory_growth > 0:
            memory_score += min(30.0, memory_growth * 10)

        scores['memory'] = min(100.0, memory_score)

        # GPU score
        gpu_util = metrics.get('gpu_util_avg', 0)
        gpu_memory = metrics.get('gpu_memory_avg_pct', 0)
        scores['gpu'] = max(gpu_util, gpu_memory)

        # I/O score
        io_total = metrics.get('io_total_avg', 0)
        # Normalize I/O (assuming 100MB/s = high load)
        scores['io'] = min(100.0, (io_total / (100 * 1024 * 1024)) * 100)

        return scores

    def _classify_performance(self, metrics: Dict[str, float],
                              scores: Dict[str, float]) -> Tuple[PerformanceTag, List[PerformanceTag], float]:
        """Classify performance type"""

        # Check for idle
        if all(score < self.thresholds['idle'] for score in scores.values()):
            return PerformanceTag.IDLE, [], 1.0

        # Sort by descending scores
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary_bottleneck = sorted_scores[0]

        # Determine primary tag
        tag_mapping = {
            'cpu': PerformanceTag.CPU_BOUND,
            'memory': PerformanceTag.MEMORY_BOUND,
            'gpu': PerformanceTag.GPU_BOUND,
            'io': PerformanceTag.IO_BOUND
        }

        primary_tag = tag_mapping.get(primary_bottleneck[0], PerformanceTag.BALANCED)

        # Determine secondary tags
        secondary_tags = []
        high_threshold = max(50.0, primary_bottleneck[1] * 0.7)

        # TODO more efficient sort?
        for resource, score in sorted_scores[1:]:
            if score > high_threshold:
                secondary_tags.append(tag_mapping[resource])

        # Compute confidence
        confidence = self._calculate_confidence(sorted_scores)

        # Check for a balanced load
        if len(secondary_tags) >= 2 and confidence < 0.6:
            primary_tag = PerformanceTag.BALANCED
            secondary_tags = [tag_mapping[s[0]] for s in sorted_scores[:3] if s[1] > 30]

        return primary_tag, secondary_tags, confidence

    @staticmethod
    def _calculate_confidence(sorted_scores: List[Tuple[str, float]]) -> float:
        """Calculate confidence in classification"""
        if len(sorted_scores) < 2:
            return 1.0

        primary_score = sorted_scores[0][1]
        secondary_score = sorted_scores[1][1]

        if primary_score == 0:
            return 0.5

        # The larger the gap, the higher the confidence
        confidence = min(1.0, (primary_score - secondary_score) / primary_score)

        # Minimum confidence is 0.3
        return max(0.3, confidence)
