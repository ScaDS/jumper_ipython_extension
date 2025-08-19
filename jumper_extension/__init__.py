import logging.config

from .logging_config import LOGGING
from .magics import load_ipython_extension, unload_ipython_extension

# Initialize logging configuration
logging.config.dictConfig(LOGGING)

__all__ = ["load_ipython_extension", "unload_ipython_extension"]
