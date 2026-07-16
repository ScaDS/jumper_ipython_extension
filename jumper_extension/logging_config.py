import logging
import os
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Use JUMPER_LOG_DIR environment variable, defaulting to home directory
BASE_LOGGING_DIR = Path(os.environ.get("JUMPER_LOG_DIR", Path.home()))
# Create a timestamped subdirectory for this session
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOGGING_DIR = BASE_LOGGING_DIR / f"jumper_logs_{timestamp}"
os.makedirs(LOGGING_DIR, exist_ok=True)


class IgnoreErrorFilter(logging.Filter):
    def filter(self, record):
        return record.levelno < logging.ERROR


class JumperExtensionOnlyFilter(logging.Filter):
    def filter(self, record):
        return "jumper_extension" in record.pathname


LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "[{levelname[0]} {asctime} {name}] {message}",
            "style": "{",
        },
    },
    "handlers": {
        "info_file": {
            "level": "INFO",
            "class": "logging.FileHandler",
            "filename": os.path.join(LOGGING_DIR, "info.log"),
            "formatter": "verbose",
        },
        "debug_file": {
            "level": "DEBUG",
            "class": "logging.FileHandler",
            "filename": os.path.join(LOGGING_DIR, "debug.log"),
            "formatter": "verbose",
        },
        "error_file": {
            "level": "ERROR",
            "class": "logging.FileHandler",
            "filename": os.path.join(LOGGING_DIR, "error.log"),
            "formatter": "verbose",
        },
        "ai_prompts_file": {
            "level": "DEBUG",
            "class": "logging.FileHandler",
            "filename": os.path.join(LOGGING_DIR, "ai_prompts.log"),
            "formatter": "verbose",
            # Created on the first prompt actually logged, not on import.
            "delay": True,
        },
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "filters": [
                "ignore_error_filter",
                "jumper_extension_only_filter",
            ],
        },
    },
    "filters": {
        "ignore_error_filter": {"()": IgnoreErrorFilter},
        "jumper_extension_only_filter": {"()": JumperExtensionOnlyFilter},
    },
    "root": {
        "handlers": [],
        "level": "WARNING",
    },
    "loggers": {
        "extension": {
            "handlers": ["console", "debug_file", "info_file", "error_file"],
            "level": "INFO",
            "propagate": True,
        },
        # Full LLM prompts and replies of the AI reviewer. Deliberately carries
        # no level of its own: it inherits "extension"'s, so raising that to
        # DEBUG turns prompt logging on. propagate is off to keep these out of
        # the console and debug.log - a single prompt runs to hundreds of lines,
        # and the console handler would put every one of them in the notebook.
        "extension.ai_prompts": {
            "handlers": ["ai_prompts_file"],
            "propagate": False,
        },
    },
}
