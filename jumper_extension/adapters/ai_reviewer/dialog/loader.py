from pathlib import Path

import yaml

from jumper_extension.adapters.ai_reviewer.dialog.models import Dialog, Option

_SPECS_DIR = Path(__file__).parent / "specs"


def load_dialog(dialog_id: str, specs_dir: Path = _SPECS_DIR) -> Dialog:
    return _build_dialog(yaml.safe_load((specs_dir / f"{dialog_id}.yaml").read_text()))


def load_all(specs_dir: Path = _SPECS_DIR) -> dict[str, Dialog]:
    return {
        path.stem: _build_dialog(yaml.safe_load(path.read_text()))
        for path in sorted(specs_dir.glob("*.yaml"))
    }


def _build_dialog(data: dict) -> Dialog:
    options = tuple(
        Option(
            id=entry["id"],
            label=entry["label"],
            effect=entry.get("effect") or {},
            description=entry.get("description", ""),
            default=bool(entry.get("default", False)),
            require_note=bool(entry.get("require_note", False)),
        )
        for entry in data["options"]
    )
    return Dialog(
        id=data["id"],
        header=data["header"],
        options=options,
        kind=data.get("kind", "single"),
        allow_note=bool(data.get("allow_note", False)),
    )
