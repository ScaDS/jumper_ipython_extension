# Python
from dataclasses import dataclass, field, asdict
from enum import Enum
from threading import RLock
from typing import Any, Optional, Tuple, Dict


class MonitorState(Enum):
    stopped = "stopped"
    running = "running"


class ReporterState(Enum):
    enabled = "enabled"
    disabled = "disabled"


class VisualizerState(Enum):
    enabled = "enabled"
    disabled = "disabled"


@dataclass
class Runtime:
    # Performance monitoring
    monitor_state: MonitorState = MonitorState.stopped
    monitor_started_at: Optional[float] = None
    monitor_stopped_at: Optional[float] = None
    # Performance reports
    report_state: ReporterState = ReporterState.disabled
    # Visualization
    visualizer_state: VisualizerState = VisualizerState.disabled


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
class Transient:
    skip_report_for_current_cell: bool = False
    last_report: Optional[Dict[str, Any]] = None   # level/format/timestamp
    last_cell_range: Optional[Tuple[int, int]] = None


@dataclass
class ExtensionState:
    version: int = 1
    runtime: Runtime = field(default_factory=Runtime)
    settings: UserSettings = field(default_factory=UserSettings)
    tmp: Transient = field(default_factory=Transient)
    _lock: RLock = field(default_factory=RLock, repr=False)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            # Exclude non-serializable parts if needed
            d = asdict(self)
            d.pop("_lock", None)
            return d

    def mark_monitor_running(self, interval: Optional[float], started_at: float):
        with self._lock:
            if self.runtime.monitor_state == MonitorState.running:
                return False
            self.settings.user_interval = interval or self.settings.default_interval
            self.runtime.monitor_state = MonitorState.running
            self.runtime.monitor_started_at = started_at
            return True

    def mark_monitor_stopped(self, stopped_at: float):
        with self._lock:
            self.runtime.monitor_state = MonitorState.stopped
            self.runtime.monitor_stopped_at = stopped_at