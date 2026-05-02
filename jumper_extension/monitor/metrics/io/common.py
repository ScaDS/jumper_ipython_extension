from jumper_extension.monitor.metrics.context import CollectionContext


class IoBackend:
    """Backend for I/O metrics."""

    name = "io-base"

    def __init__(self):
        pass

    def setup(self) -> None:
        return None

    def snapshot(self, context: CollectionContext) -> None:
        return None

    def collect(self, level: str, context: CollectionContext) -> list[int]:
        raise NotImplementedError
