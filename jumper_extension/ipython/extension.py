import logging

from jumper_extension.core.messages import (
    ExtensionInfoCode,
    EXTENSION_INFO_MESSAGES,
)
from jumper_extension.core.service import build_perfmonitor_service
from jumper_extension.ipython.magics import PerfmonitorMagics


logger = logging.getLogger("extension")
_perfmonitor_magics = None


def load_ipython_extension(ipython):
    global _perfmonitor_magics
    service = build_perfmonitor_service()
    _perfmonitor_magics = PerfmonitorMagics(ipython, service)
    ipython.events.register("pre_run_cell", _perfmonitor_magics.pre_run_cell)
    ipython.events.register("post_run_cell", _perfmonitor_magics.post_run_cell)
    ipython.register_magics(_perfmonitor_magics)
    logger.info(EXTENSION_INFO_MESSAGES[ExtensionInfoCode.EXTENSION_LOADED])


def unload_ipython_extension(ipython):
    global _perfmonitor_magics
    if _perfmonitor_magics:
        ipython.events.unregister(
            "pre_run_cell", _perfmonitor_magics.pre_run_cell
        )
        ipython.events.unregister(
            "post_run_cell", _perfmonitor_magics.post_run_cell
        )
        _perfmonitor_magics.service.close()
        _perfmonitor_magics = None