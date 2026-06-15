"""Hydra-like composition of the application config from config.yaml `defaults:`."""

from importlib import resources

import yaml

from jumper_extension.config.models import AppConfig


def _read_yaml(package: str, filename: str) -> dict:
    try:
        text = resources.files(package).joinpath(filename).read_text(encoding="utf-8")
    except AttributeError:
        text = resources.read_text(package, filename, encoding="utf-8")
    return yaml.safe_load(text) or {}


def _compose() -> AppConfig:
    root = _read_yaml("jumper_extension.config", "config.yaml")
    merged: dict = {}
    for entry in root.get("defaults", []):
        for group, option in entry.items():
            package = "jumper_extension.config." + group.replace("/", ".")
            data = _read_yaml(package, f"{option}.yaml")
            *parents, key = group.split("/")
            target = merged
            for parent in parents:
                target = target.setdefault(parent, {})
            target[key] = data
    return AppConfig.model_validate(merged)


# Loaded once at import time — not affected by test patches on builtins.open.
_APP_CONFIG: AppConfig = _compose()


def load_config() -> AppConfig:
    """Return the application config composed at import time."""
    return _APP_CONFIG
