from pathlib import Path

import yaml


def load_c_collectors_config() -> list[str]:
    """Return the ordered list of C collector names from collectors.yaml."""
    path = Path(__file__).parent / "collectors.yaml"
    with path.open() as f:
        return yaml.safe_load(f)["collectors"]
