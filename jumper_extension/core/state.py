# Python
from dataclasses import dataclass, field, asdict
from enum import Enum
from threading import RLock
from typing import Any, Optional, Tuple, Dict


@dataclass
class ExportVars:
    perfdata: str = "perfdata_df"
    cell_history: str = "cell_history_df"


@dataclass
class PerfomanceReports:
    enabled: bool = False
    level: str = "process"
    text: bool = False


@dataclass
class UserSettings:
    perfreports: PerfomanceReports = field(default_factory=PerfomanceReports)
    default_interval: float = 1.0
    user_interval: Optional[float] = None
    export_vars: ExportVars = field(default_factory=ExportVars)
