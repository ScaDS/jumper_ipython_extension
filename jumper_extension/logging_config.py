import logging
import os
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Use JUMPER_LOG_DIR environment variable, defaulting to home directory
BASE_LOGGING_DIR = Path(os.environ.get("JUMPER_LOG_DIR", Path.home()))
# Named per session; created only once something is actually logged into it.
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOGGING_DIR = BASE_LOGGING_DIR / f"jumper_logs_{timestamp}"


class LazyFileHandler(logging.FileHandler):
    """A FileHandler that creates its directory when the first record lands.

    Paired with ``delay``, this keeps a bare ``import jumper_extension`` - or a
    benchmark replay that logs nothing - from leaving behind a directory of
    empty files.
    """

    def _open(self):
        os.makedirs(os.path.dirname(self.baseFilename), exist_ok=True)
        return super()._open()


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
            "class": "jumper_extension.logging_config.LazyFileHandler",
            "filename": os.path.join(LOGGING_DIR, "info.log"),
            "formatter": "verbose",
            "delay": True,
        },
        "debug_file": {
            "level": "DEBUG",
            "class": "jumper_extension.logging_config.LazyFileHandler",
            "filename": os.path.join(LOGGING_DIR, "debug.log"),
            "formatter": "verbose",
            "delay": True,
        },
        "error_file": {
            "level": "ERROR",
            "class": "jumper_extension.logging_config.LazyFileHandler",
            "filename": os.path.join(LOGGING_DIR, "error.log"),
            "formatter": "verbose",
            "delay": True,
        },
        "ai_prompts_file": {
            "level": "DEBUG",
            "class": "jumper_extension.logging_config.LazyFileHandler",
            "filename": os.path.join(LOGGING_DIR, "ai_prompts.log"),
            "formatter": "verbose",
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
            "level": "DEBUG",
            "propagate": True,
        },
        # No level of its own: inherits "extension"'s, so raising that to DEBUG
        # turns prompt logging on. propagate is off to keep hundreds of prompt
        # lines out of the notebook console.
        "extension.ai_prompts": {
            "handlers": ["ai_prompts_file"],
            "propagate": False,
        },
    },
}
