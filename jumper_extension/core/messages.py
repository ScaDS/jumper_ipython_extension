"""Message codes and templates used by the JUmPER extension.

This module defines enums for error and info codes, maps them to
human-readable message templates, and exposes helpers for working with
those messages.
"""

from enum import Enum, auto

from jumper_extension.logging_config import LOGGING

MESSAGE_PREFIX = "[JUmPER]"


class ExtensionErrorCode(Enum):
    """Error codes emitted by the extension."""

    PYNVML_NOT_AVAILABLE = auto()
    NVIDIA_DRIVERS_NOT_AVAILABLE = auto()
    ADLX_NOT_AVAILABLE = auto()
    AMD_DRIVERS_NOT_AVAILABLE = auto()
    NO_PERFORMANCE_DATA = auto()
    INVALID_CELL_RANGE = auto()
    INVALID_INTERVAL_VALUE = auto()
    INVALID_METRIC_SUBSET = auto()
    NO_ACTIVE_MONITOR = auto()
    MONITOR_ALREADY_RUNNING = auto()
    UNSUPPORTED_FORMAT = auto()
    INVALID_LEVEL = auto()
    DEFINE_LEVEL = auto()
    NO_CELL_HISTORY = auto()


class ExtensionInfoCode(Enum):
    """Informational codes emitted by the extension."""

    IMPRECISE_INTERVAL = auto()
    MISSED_MEASUREMENTS = auto()
    PERFORMANCE_REPORTS_DISABLED = auto()
    EXTENSION_LOADED = auto()
    PERFORMANCE_REPORTS_ENABLED = auto()
    MONITOR_STARTED = auto()
    MONITOR_STOPPED = auto()
    EXPORT_SUCCESS = auto()
    PERFORMANCE_DATA_AVAILABLE = auto()
    HTML_REPORTS_NOT_AVAILABLE = auto()
    PLOTS_NOT_AVAILABLE = auto()
    SESSION_IMPORTED = auto()
    IMPORTED_SESSION_PLOT = auto()
    IMPORTED_SESSION_RESOURCES = auto()

_BASE_EXTENSION_ERROR_MESSAGES = {
    ExtensionErrorCode.PYNVML_NOT_AVAILABLE: (
        "Pynvml not available. GPU monitoring disabled."
    ),
    ExtensionErrorCode.NVIDIA_DRIVERS_NOT_AVAILABLE: (
        "NVIDIA drivers not available. NVIDIA GPU monitoring disabled."
    ),
    ExtensionErrorCode.ADLX_NOT_AVAILABLE: (
        "ADLXPybind not available. AMD GPU monitoring disabled."
    ),
    ExtensionErrorCode.AMD_DRIVERS_NOT_AVAILABLE: (
        "AMD drivers not available. AMD GPU monitoring disabled."
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
    ExtensionErrorCode.UNSUPPORTED_FORMAT: (
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
    ExtensionInfoCode.MISSED_MEASUREMENTS: (
        "Missed measurements: ({perc_missed_measurements: .2f})"
    ),
    ExtensionInfoCode.PERFORMANCE_REPORTS_DISABLED: (
        "Performance reports for each cell disabled"
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
    ExtensionInfoCode.PERFORMANCE_DATA_AVAILABLE: (
        "Performance data DataFrame available as '{var_name}'"
    ),
    ExtensionInfoCode.HTML_REPORTS_NOT_AVAILABLE: (
        "HTML reports are not available: {reason}"
    ),
    ExtensionInfoCode.PLOTS_NOT_AVAILABLE: (
        "Plots are not available: {reason}"
    ),
    ExtensionInfoCode.SESSION_IMPORTED: ("Session imported successfully: {source}"),
    ExtensionInfoCode.IMPORTED_SESSION_PLOT: ("Using imported session data for plotting: {source}"),
    ExtensionInfoCode.IMPORTED_SESSION_RESOURCES: ("Showing resources from imported session: {source}"),
}

def _apply_prefix(messages):
    """Attach the standard JUmPER prefix to all message templates.

    Args:
        messages: Mapping from message code enum to message template.

    Returns:
        dict: New mapping with the message prefix added to each
        template.
    """
    return {
        code: f"{MESSAGE_PREFIX}: {text}" for code, text in messages.items()
    }


EXTENSION_ERROR_MESSAGES = _apply_prefix(_BASE_EXTENSION_ERROR_MESSAGES)
EXTENSION_INFO_MESSAGES = _apply_prefix(_BASE_EXTENSION_INFO_MESSAGES)


def get_jumper_process_error_hint() -> str:
    """Return a hint pointing to the error log file.

    Returns:
        str: Human-readable hint with the path to the error log file.
    """
    jumper_process_error_hint = (
        "\nHint: full error info saved to log file: "
        f"{LOGGING['handlers']['error_file']['filename']}"
    )
    return jumper_process_error_hint
