import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger("extension")


class PerformanceTag(Enum):
    """Performance tags for classifying cells"""
    NORMAL = "normal"
    CPU_BOUND = "cpu-bound"
    MEMORY_BOUND = "memory-bound"
    GPU_UTIL_BOUND = "gpu-util-bound"
    GPU_MEMORY_BOUND = "gpu-memory-bound"
    GPU_ALLOCATED_BUT_NOT_USED = "gpu-allocated-but-not-used"

    def __str__(self):
        return self.value


@dataclass
class TagScore:
    """Tag with its score for ranking"""
    tag: PerformanceTag
    score: float


class PerformanceAnalyzer:
    """
    Performance analyzer to determine workload type using relative thresholds.

    Inspired by `JobLabeller` from:
    https://gitlab.hrz.tu-chemnitz.de/pika/pika-server/-/blob/
    619d62926cd85f8c20589c75aba0c6e2c51087e1/
    src/post_processing/post_processing.py#L711
    """
    # Default thresholds
    DEFAULT_THRESHOLDS = {
        'memory_ratio': 0.8,  # memory limit 0.80
        'cpu_ratio': 0.7,  # CPU capacity 0.70
        'gpu_util_ratio': 0.8,  # GPU utilization
        'gpu_memory_ratio': 0.8,  # GPU memory

        # --- thresholds for "GPU idle" detection
        # minimum memory usage required to treat GPU as allocated
        'gpu_alloc_min_mem_gb': 0.1,
        # minimum GPU utilization to treat GPU in idle state
        'gpu_util_idle_threshold': 0.05,
        # minimum overall usage fraction required to trigger the tag
        'gpu_alloc_min_fraction': 0.5,

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

    def analyze_cell_performance(
        self,
        perfdata,
        memory_limit: float,
        gpu_memory_limit: Optional[float] = None
    ) -> List[TagScore]:
        """
        Analyze cell performance and determine tags

        Args:
            perfdata: DataFrame with performance data
            memory_limit: System memory limit in GB
            gpu_memory_limit: GPU memory limit in GB (if available)

        Returns:
            List[TagScore]: Ranked performance tags for the cell
        """

        logger.debug(f"{memory_limit = }")
        logger.debug(f"{gpu_memory_limit = }")

        # Compute normalized metrics
        metrics = self._compute_metrics(perfdata, gpu_memory_limit)

        # Calculate resource utilization ratios
        ratios = self._calculate_utilization_ratios(metrics, memory_limit, gpu_memory_limit)

        # Create the ranked tags list
        ranked_tags = self._create_ranked_tags(ratios)

        # Detect "GPU allocated but not used" and prepend if applicable
        gpu_unused_tag = self._detect_gpu_allocated_but_not_used(perfdata, gpu_memory_limit)
        if gpu_unused_tag is not None:
            # Prepend the GPU allocated but not used as this is the most important tag
            ranked_tags = [gpu_unused_tag] +  ranked_tags

        logger.debug(f"{ranked_tags = }")

        return ranked_tags if ranked_tags else [TagScore(PerformanceTag.NORMAL, 0.0)]

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

        logger.debug(f"ratios: {ratios}")

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

        # Sort by descending ratios
        sorted_ratios = sorted(ratios.items(), key=lambda x: x[1], reverse=True)

        # Create ranked tags for resources that exceed the minimum threshold
        tag_mapping = {
            'cpu': PerformanceTag.CPU_BOUND,
            'memory': PerformanceTag.MEMORY_BOUND,
            'gpu_util': PerformanceTag.GPU_UTIL_BOUND,
            'gpu_memory': PerformanceTag.GPU_MEMORY_BOUND,
        }

        ranked_tags = []

        for resource, ratio in sorted_ratios:
            threshold_key = f'{resource}_ratio'
            threshold = self.thresholds.get(threshold_key, 0.0)
            if ratio >= threshold:
                tag = tag_mapping.get(resource)
                if tag:
                    ranked_tags.append(TagScore(tag, ratio))

        return ranked_tags

    def _detect_gpu_allocated_but_not_used(
        self,
        perfdata,
        gpu_memory_limit: Optional[float],
    ) -> Optional[TagScore]:
        """
        Detect case when GPU memory is allocated but GPU compute utilization stays idle
        for a significant fraction of measurement time.
        """
        # must have GPU columns and a GPU present
        if gpu_memory_limit is None:
            return None
        if 'gpu_mem_avg' not in perfdata.columns or 'gpu_util_avg' not in perfdata.columns:
            return None
        if perfdata.empty:
            return None

        memory_threshold_gb = max(float(self.thresholds.get('gpu_alloc_min_mem_gb', 0.1)), 0.0)
        utilization_idle_threshold = float(self.thresholds.get('gpu_util_idle_threshold', 0.05))  # 0..1
        min_fraction = float(self.thresholds.get('gpu_alloc_min_fraction', 0.5))        # 0..1

        # allocation considered if memory usage exceeds memory_threshold_gb
        mask_alloc = perfdata['gpu_mem_avg'] > memory_threshold_gb
        if mask_alloc.sum() == 0:
            return None

        # idle if util ≤ util_idle_thr * 100 (%)
        mask_idle = perfdata['gpu_util_avg'] <= (utilization_idle_threshold * 100.0)

        mask_alloc_and_idle = mask_alloc & mask_idle
        frac = float(mask_alloc_and_idle.mean())

        logger.debug(f"GPU idle check:")
        logger.debug(f"gpu_mem_avg:\n{perfdata['gpu_mem_avg']}")
        logger.debug(f"mask_alloc:\n{mask_alloc}\n")
        logger.debug(f"gpu_util_avg:\n{perfdata['gpu_util_avg']}")
        logger.debug(f"mask_idle:\n{mask_idle}\n")
        logger.debug(f"{min_fraction = }")
        logger.debug(f"{frac = }")

        if frac >= min_fraction:
            return TagScore(PerformanceTag.GPU_ALLOCATED_BUT_NOT_USED, frac)
        return None

