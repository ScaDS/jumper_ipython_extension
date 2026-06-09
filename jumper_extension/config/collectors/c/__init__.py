from importlib import resources

import yaml


def load_c_collectors_config() -> list[str]:
    """Return the ordered list of C collector names from collectors.yaml."""
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
    return yaml.safe_load(config_text)["collectors"]
