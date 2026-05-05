"""Public interfaces for building custom metric collectors.

To add a new metric source implement a :class:`CollectorBackend` subclass and a
matching :class:`StorageHandler`, then register them in ``collectors.yaml``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol

from jumper_extension.monitor.metrics.context import CollectionContext

__all__ = ["CollectionContext", "CollectorBackend", "StorageHandler"]


class CollectorBackend(ABC):
    """Abstract base for all metric collector backends.

    A backend collects one category of metrics (CPU, memory, GPU, I/O, or
    process state).  The pipeline calls the three lifecycle methods on every
    backend once per tick:

        setup()             — once at startup
        snapshot(context)   — once per tick, before any collect()
        collect(level, ctx) — once per tick per active level

    Subclass contract:
        - Define a unique ``name`` class attribute, e.g. ``"cpu-psutil"``.
        - Implement :meth:`collect`.  Override :meth:`setup` and
          :meth:`snapshot` only when initialization or per-tick
          pre-computation is needed.
        - Pair with a :class:`StorageHandler` that converts the value
          returned by :meth:`collect` into a flat ``dict[str, float]``.

    Minimal example::

        class MyBackend(CollectorBackend):
            name = "my-metric"

            def collect(self, level: str, context: CollectionContext):
                return 42.0  # pair with ScalarHandler(column="my_value")
    """

    name: str

    def setup(self) -> dict | None:
        """Initialize resources.  Called once before collection starts.

        Override to acquire handles, open connections, or discover hardware.

        Returns:
            Optional metadata dict.  GPU backends use this to report
            ``{"num_gpus": n, "gpu_memory": f, "gpu_name": s}`` so the
            pipeline can populate :class:`NodeInfo` before deferred backends
            are built.  Return ``None`` (or omit the override) otherwise.
        """
        return None

    def snapshot(self, context: CollectionContext) -> None:
        """Pre-tick snapshot: populate shared context before collect().

        The pipeline calls ``snapshot()`` on *all* backends before calling
        ``collect()`` on any of them, so data written here is visible to
        every ``collect()`` call in the same tick.

        The process backend uses this to enumerate live PIDs and fill
        ``context`` with per-PID cpu/rss/io counters so that CPU, memory, and
        I/O backends can read them without redundant syscalls.

        Override only if your backend needs to pre-compute shared state.
        """
        return None

    @abstractmethod
    def collect(self, level: str, context: CollectionContext) -> Any:
        """Collect one sample for the given aggregation level.

        Args:
            level: Aggregation scope.  One of ``"system"``, ``"process"``,
                   ``"user"``, or ``"slurm"`` (see :class:`StorageHandler`
                   for full semantics).
            context: Shared state populated by :meth:`snapshot` earlier in
                     this tick.  Contains live PIDs and per-PID cpu/rss/io
                     snapshots.

        Returns:
            Raw sample value passed directly to the paired
            :class:`StorageHandler`.  The type must match what the handler
            expects — see the built-in handlers for standard pairings.
        """
        ...


class StorageHandler(Protocol):
    def transform(self, raw, level: str) -> dict[str, float]:
        """Return a flat column→value dict for one sample.

        Args:
            raw: The value returned by the paired :class:`CollectorBackend`.
                 Intentionally untyped — each handler is coupled to exactly
                 one collector and knows the concrete type it expects
                 (``float``, ``list[float]``, ``list[int]``,
                 ``tuple[list[float], ...]``, or ``None``).  See the
                 built-in handlers for examples of each pairing.
            level: Aggregation scope for this sample.  One of:

                 ``"system"``  — all processes on the machine,
                 ``"process"`` — the monitored process and its children,
                 ``"user"``    — all processes owned by the current user,
                 ``"slurm"``   — all processes belonging to the current
                                 Slurm job (only present when running inside
                                 a Slurm allocation).

                 Stateful handlers (e.g. rate computation) use ``level`` as
                 a key to keep per-scope state separate.

        Returns:
            A flat ``{column_name: value}`` dict.  An empty dict is valid
            and means this handler contributes no columns for this sample.
        """
        ...
