"""Mini-Hydra: _target_-based instantiation without the full Hydra dependency."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any

import yaml


def _read_collectors_config() -> dict:
    config_path = Path(__file__).parent / "collectors.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# Loaded once at import time — not affected by test patches on builtins.open.
_COLLECTORS_CONFIG: dict = _read_collectors_config()


def load_collectors_config() -> dict:
    """Return the collectors config loaded at import time."""
    return _COLLECTORS_CONFIG


def instantiate_backend(cls: type, available: dict) -> Any:
    """Instantiate *cls* injecting only the kwargs it declares by name.

    Each parameter name in ``cls.__init__`` is looked up in *available*;
    unrecognised names are silently ignored.
    """
    sig = inspect.signature(cls.__init__)
    filtered = {k: v for k, v in available.items() if k in sig.parameters}
    return cls(**filtered)


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
