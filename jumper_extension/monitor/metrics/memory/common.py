from jumper_extension.adapters.data import NodeInfo
from jumper_extension.monitor.metrics.context import CollectionContext


class MemoryBackend:
    """Backend for memory metrics."""

    name = "memory-base"

    def __init__(self, node_info: NodeInfo):
        self._node_info = node_info

    def setup(self) -> None:
        return None

    def snapshot(self, context: CollectionContext) -> None:
        return None

    def collect(self, level: str, context: CollectionContext) -> float:
        raise NotImplementedError
