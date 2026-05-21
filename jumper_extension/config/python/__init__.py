from pathlib import Path

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
