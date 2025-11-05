import copy
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExportVars:
    perfdata: str = "perfdata_df"
    cell_history: str = "cell_history_df"

@dataclass
class LoadedVars:
    perfdata: str = "loaded_perfdata_df"
    cell_history: str = "loaded_cell_history_df"


@dataclass
class PerfomanceReports:
    enabled: bool = False
    level: str = "process"
    text: bool = False


@dataclass
class PerformanceMonitoring:
    default_interval: float = 1.0
    user_interval: Optional[float] = None
    running: bool = False


@dataclass
class Settings:
    perfreports: PerfomanceReports = field(default_factory=PerfomanceReports)
    monitoring: PerformanceMonitoring = field(default_factory=PerformanceMonitoring)
    export_vars: ExportVars = field(default_factory=ExportVars)
    loaded_vars: LoadedVars = field(default_factory=LoadedVars)

    def snapshot(self) -> "Settings":
        return copy.deepcopy(self)

