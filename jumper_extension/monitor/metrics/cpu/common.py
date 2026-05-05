from jumper_extension.adapters.data import NodeInfo
from jumper_extension.monitor.metrics.context import CollectionContext


class CpuCollectorBackend:
    """Backend for CPU metrics."""

    name = "cpu-base"

    def __init__(self, node_info: NodeInfo):
        self._node_info = node_info

    def setup(self) -> None:
        return None

    def snapshot(self, context: CollectionContext) -> None:
        return None

    def collect(self, level: str, context: CollectionContext) -> list[float]:
        raise NotImplementedError
