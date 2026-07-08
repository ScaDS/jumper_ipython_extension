import dataclasses


@dataclasses.dataclass(frozen=True)
class Option:
    """One selectable choice in a dialog.

    ``effect`` is an opaque payload interpreted by a domain handler (for the
    optimization dialogs it holds ``{"overrides": {id: enabled}}``), which keeps
    the dialog schema decoupled from whatever the choice controls.
    """
    id: str
    label: str
    effect: dict = dataclasses.field(default_factory=dict)
    description: str = ""
    default: bool = False
    require_note: bool = False


@dataclasses.dataclass(frozen=True)
class Dialog:
    """A single question with its options.

    Identical whether loaded from ``specs/*.yaml`` (deterministic) or built by
    the agent at runtime (dynamic) - the presenter and handlers treat both the
    same way.
    """
    id: str
    header: str
    options: tuple[Option, ...]
    kind: str = "single"          # single | multi | text
    allow_note: bool = False

    def default_option(self) -> Option:
        for option in self.options:
            if option.default:
                return option
        return self.options[0]


@dataclasses.dataclass(frozen=True)
class Answer:
    dialog_id: str
    selected: tuple[str, ...]
    note: str | None = None

    @property
    def option_id(self) -> str:
        return self.selected[0]
