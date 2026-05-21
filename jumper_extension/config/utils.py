"""Mini-Hydra: _target_-based instantiation without the full Hydra dependency."""

from __future__ import annotations

import importlib
from typing import Any


def instantiate(cfg: dict, **injected: Any) -> Any:
    """Instantiate a class from a config dict with a ``_target_`` key.

    ``_target_`` is popped; remaining keys are forwarded as keyword arguments
    together with *injected* values declared via ``inject:`` in the config.
    """
    cfg = dict(cfg)
    target = cfg.pop("_target_")
    module_path, class_name = target.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(**cfg, **injected)
