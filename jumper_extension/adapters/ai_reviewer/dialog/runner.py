from jumper_extension.adapters.ai_reviewer.dialog.models import Answer, Dialog


class DialogPresenter:
    """Renders a dialog in the chat and returns the user's answer.

    Concrete presenters (notebook widget, terminal, ...) implement ``present``.
    Everything else depends only on this interface, so the presentation
    technology can change without touching dialog specs, the schema, or the
    effect handlers.
    """

    def present(self, dialog: Dialog) -> Answer:
        raise NotImplementedError


class DefaultPresenter(DialogPresenter):
    """Non-interactive fallback: selects each dialog's default option.

    Used when ``interactive: false``, in tests, and until a real widget
    presenter is wired in.
    """

    def present(self, dialog: Dialog) -> Answer:
        option = dialog.default_option()
        return Answer(dialog_id=dialog.id, selected=(option.id,))


def render_menu(
    dialog: Dialog,
    cursor: int,
    selected: set[str],
    note: str | None = None,
) -> str:
    """Text of the arrow-select menu; ``❯`` marks the cursor row.

    Reused by any concrete presenter regardless of the widget technology.
    """
    lines = [f"{dialog.header}:"]
    for index, option in enumerate(dialog.options):
        pointer = "❯" if index == cursor else " "
        if dialog.kind == "multi":
            mark = "[x]" if option.id in selected else "[ ]"
        else:
            mark = "●" if option.id in selected else "○"
        lines.append(f"{pointer} {mark} {option.label}")
    if note:
        lines.append(f"      note: {note}")
    return "\n".join(lines)


def resolve_overrides(dialog: Dialog, answer: Answer) -> dict:
    """Merge the ``overrides`` payloads of the selected options into one map."""
    by_id = {option.id: option for option in dialog.options}
    overrides = {}
    for option_id in answer.selected:
        overrides.update(by_id[option_id].effect.get("overrides", {}))
    return overrides
