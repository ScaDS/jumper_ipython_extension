from enum import Enum, auto

from .logging_config import LOGGING

MESSAGE_PREFIX = "[JUmPER]"


class ExtensionErrorCode(Enum):
    PYNVML_NOT_AVAILABLE = auto()
    NVIDIA_DRIVERS_NOT_AVAILABLE = auto()
    NO_PERFORMANCE_DATA = auto()
    INVALID_CELL_RANGE = auto()
    INVALID_INTERVAL_VALUE = auto()
    INVALID_METRIC_SUBSET = auto()
    NO_ACTIVE_MONITOR = auto()
    MONITOR_ALREADY_RUNNING = auto()
    UNSUPPORTED_EXPORT_FORMAT = auto()
    INVALID_LEVEL = auto()
    DEFINE_LEVEL = auto()
    NO_CELL_HISTORY = auto()


class ExtensionInfoCode(Enum):
    IMPRECISE_INTERVAL = auto()
    PERFORMANCE_REPORTS_DISABLED = auto()
    EXTENSION_LOADED = auto()
    PERFORMANCE_REPORTS_ENABLED = auto()
    MONITOR_STARTED = auto()
    MONITOR_STOPPED = auto()
    EXPORT_SUCCESS = auto()


_BASE_EXTENSION_ERROR_MESSAGES = {
    ExtensionErrorCode.PYNVML_NOT_AVAILABLE: (
        "Pynvml not available. GPU monitoring disabled."
    ),
    ExtensionErrorCode.NVIDIA_DRIVERS_NOT_AVAILABLE: (
        "NVIDIA drivers not available. GPU monitoring disabled."
    ),
    ExtensionErrorCode.NO_PERFORMANCE_DATA: (
        "No performance data available or recorded cells are too short"
    ),
    ExtensionErrorCode.INVALID_CELL_RANGE: (
        "Invalid cell range format: {cell_range}"
    ),
    ExtensionErrorCode.INVALID_INTERVAL_VALUE: (
        "Invalid interval value: {interval}"
    ),
    ExtensionErrorCode.INVALID_METRIC_SUBSET: (
        "Unknown metric subset: {subset}. Supported subsets: "
        "{supported_subsets}"
    ),
    ExtensionErrorCode.NO_ACTIVE_MONITOR: (
        "No active performance monitoring session"
    ),
    ExtensionErrorCode.MONITOR_ALREADY_RUNNING: (
        "Performance monitoring already running"
    ),
    ExtensionErrorCode.UNSUPPORTED_EXPORT_FORMAT: (
        "Unsupported format: {format}. Supported formats: {supported_formats}"
    ),
    ExtensionErrorCode.INVALID_LEVEL: (
        "Invalid level: {level}. Available levels: {levels}"
    ),
    ExtensionErrorCode.DEFINE_LEVEL: (
        "Please define performance measurement level with --level argument. "
        "Available levels: {levels}"
    ),
    ExtensionErrorCode.NO_CELL_HISTORY: ("No cell history available"),
}


_BASE_EXTENSION_INFO_MESSAGES = {
    ExtensionInfoCode.IMPRECISE_INTERVAL: (
        "Measurements might not meet the desired interval ({interval}s) "
        "due to performance constraints"
    ),
    ExtensionInfoCode.PERFORMANCE_REPORTS_DISABLED: (
        "Performance reports for each celldisabled"
    ),
    ExtensionInfoCode.PERFORMANCE_REPORTS_ENABLED: (
        "Performance reports enabled for each cell ({options_message})"
    ),
    ExtensionInfoCode.EXTENSION_LOADED: ("Perfmonitor extension loaded"),
    ExtensionInfoCode.MONITOR_STARTED: (
        "Performance monitoring started (PID: {pid}, Interval: {interval}s)"
    ),
    ExtensionInfoCode.MONITOR_STOPPED: (
        "Performance monitoring stopped (ran for {seconds:.2f} seconds)"
    ),
    ExtensionInfoCode.EXPORT_SUCCESS: ("Exported to {filename}"),
}


def _apply_prefix(messages):
    return {
        code: f"{MESSAGE_PREFIX}: {text}" for code, text in messages.items()
    }


EXTENSION_ERROR_MESSAGES = _apply_prefix(_BASE_EXTENSION_ERROR_MESSAGES)
EXTENSION_INFO_MESSAGES = _apply_prefix(_BASE_EXTENSION_INFO_MESSAGES)


def get_jumper_process_error_hint():
    jumper_process_error_hint = (
        "\nHint: full error info saved to log file: "
        f"{LOGGING['handlers']['error_file']['filename']}"
    )
    return jumper_process_error_hint
