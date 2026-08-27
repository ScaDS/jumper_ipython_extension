"""Runtime session state for the JUmPER core.

This module defines dataclasses that hold mutable, per-session state
toggled by magic commands (automatic reports, monitoring interval and
running flag, visualizer backend). Configurable defaults for these
values live in ``jumper_extension.config`` (see ``SettingsConfig``);
``State.from_config()`` seeds the initial values from there.
"""

import copy
from dataclasses import dataclass, field
from typing import Optional

from jumper_extension.config.models import SettingsConfig

SETTINGS_DEFAULTS = SettingsConfig()


@dataclass
class PerfReportsState:
    """Current configuration of automatic per-cell performance reports.

    Attributes:
        enabled: Whether per-cell reports are enabled.
        level: Monitoring level used when generating reports.
        text: If True, use text reports instead of HTML.
    """

    enabled: bool = False
    level: str = SETTINGS_DEFAULTS.perfreports.level
    text: bool = SETTINGS_DEFAULTS.perfreports.text


@dataclass
class MonitoringState:
    """Current state of the performance monitoring loop.

    Attributes:
        user_interval: User-provided interval overriding the default.
        running: Whether monitoring is currently running.
    """

    user_interval: Optional[float] = None
    running: bool = False


@dataclass
class State:
    """Mutable runtime state for a JUmPER session.

    Attributes:
        perfreports: Current configuration of per-cell reports.
        monitoring: Current state of the monitoring loop.
        visualizer_backend: Currently selected backend used for plotting.
    """

    perfreports: PerfReportsState = field(default_factory=PerfReportsState)
    monitoring: MonitoringState = field(default_factory=MonitoringState)
    visualizer_backend: str = SETTINGS_DEFAULTS.visualizer_backend

    def snapshot(self) -> "State":
        """Return a deep copy of the current state.

        Returns:
            State: Independent copy of the current runtime state.
        """
        return copy.deepcopy(self)

    @classmethod
    def from_config(cls, cfg: SettingsConfig) -> "State":
        """Build initial session state from configured defaults.

        Args:
            cfg: Settings defaults loaded from the application config.

        Returns:
            State: A new state seeded with the configured defaults.
        """
        return cls(
            perfreports=PerfReportsState(
                level=cfg.perfreports.level,
                text=cfg.perfreports.text,
            ),
            visualizer_backend=cfg.visualizer_backend,
        )
