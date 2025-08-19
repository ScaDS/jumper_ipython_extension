import logging
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGGING_DIR = PROJECT_ROOT / "logs_jumper"
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
    },
}
