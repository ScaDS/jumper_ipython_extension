import dataclasses


@dataclasses.dataclass(frozen=True)
class Strategy:
    """A named review strategy selected via ``--strategy``.

    ``overrides`` is a flat ``id -> enabled`` map merged from the strategy's
    prompt-item toggles and context-source toggles; it is passed both to
    ``PromptLibrary.render`` (drops given/rule items) and to the context
    collector (skips disabled sources), so one strategy steers both layers.
    """
    id: str
    name: str
    description: str
    overrides: dict
    require_note: bool = False
