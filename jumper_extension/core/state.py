# Python
from dataclasses import dataclass, field, asdict
from enum import Enum
from threading import RLock
from typing import Any, Optional, Tuple, Dict


class MonitorState(Enum):
    stopped = "stopped"
    running = "running"


@dataclass
class Runtime:
    # Performance monitoring
    monitor_state: MonitorState = MonitorState.stopped
    monitor_started_at: Optional[float] = None
    monitor_stopped_at: Optional[float] = None


@dataclass
class ExportVars:
    perfdata: str = "perfdata_df"
    cell_history: str = "cell_history_df"


@dataclass
class UserSettings:
    perfreports_enabled: bool = False
    perfreports_level: str = "process"
    perfreports_text: bool = False
    default_interval: float = 1.0
    user_interval: Optional[float] = None
    export_vars: ExportVars = field(default_factory=ExportVars)


@dataclass
class ExtensionState:
    version: int = 1
    runtime: Runtime = field(default_factory=Runtime)
    settings: UserSettings = field(default_factory=UserSettings)

    def snapshot(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def mark_monitor_running(self, interval: Optional[float], started_at: float):
        if self.runtime.monitor_state == MonitorState.running:
            return False
        self.settings.user_interval = interval or self.settings.default_interval
        self.runtime.monitor_state = MonitorState.running
        self.runtime.monitor_started_at = started_at
        return True

    def mark_monitor_stopped(self, stopped_at: float):
        self.runtime.monitor_state = MonitorState.stopped
        self.runtime.monitor_stopped_at = stopped_at
