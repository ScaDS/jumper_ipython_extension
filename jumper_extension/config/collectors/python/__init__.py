from importlib import resources

import yaml


def _read_collectors_config() -> dict:
    try:
        config_text = (
            resources.files(__package__)
            .joinpath("collectors.yaml")
            .read_text(encoding="utf-8")
        )
    except AttributeError:
        config_text = resources.read_text(
            __package__,
            "collectors.yaml",
            encoding="utf-8",
        )
    return yaml.safe_load(config_text)


# Loaded once at import time — not affected by test patches on builtins.open.
_COLLECTORS_CONFIG: dict = _read_collectors_config()


def load_collectors_config() -> dict:
    """Return the collectors config loaded at import time."""
    return _COLLECTORS_CONFIG
