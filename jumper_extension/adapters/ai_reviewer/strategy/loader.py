from pathlib import Path

import yaml

from jumper_extension.adapters.ai_reviewer.strategy.models import Strategy

_STRATEGIES_PATH = Path(__file__).parent / "strategies.yaml"


def load_strategies(path: Path = _STRATEGIES_PATH) -> dict[str, Strategy]:
    """Parse ``strategies.yaml`` into ``{id: Strategy}``.

    Each strategy's ``effect.overrides`` (prompt items) and ``effect.context``
    (context sources) are merged into one flat ``id -> enabled`` map.
    """
    data = yaml.safe_load(path.read_text())
    strategies = {}
    for entry in data["strategies"]:
        effect = entry.get("effect") or {}
        overrides = {
            **(effect.get("context") or {}),
            **(effect.get("overrides") or {}),
        }
        strategies[entry["id"]] = Strategy(
            id=entry["id"],
            name=entry["name"],
            description=entry.get("description", ""),
            overrides=overrides,
            require_note=bool(entry.get("require_note", False)),
        )
    return strategies


def strategy_ids(path: Path = _STRATEGIES_PATH) -> list[str]:
    return list(load_strategies(path))


def get_strategy(strategy_id: str, path: Path = _STRATEGIES_PATH) -> Strategy:
    return load_strategies(path)[strategy_id]
