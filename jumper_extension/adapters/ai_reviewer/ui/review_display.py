import difflib
import logging
from pathlib import Path

from IPython.display import HTML, display
from jinja2 import Environment, FileSystemLoader, select_autoescape

from jumper_extension.adapters.ai_reviewer.agent.state import OptimizationState
from jumper_extension.core.messages import (
    EXTENSION_INFO_MESSAGES,
    ExtensionInfoCode,
)

logger = logging.getLogger("extension")

_NORMAL_TAG = "normal"


def _diff_lines(original: str, suggested: str) -> list[dict]:
    """Classify unified-diff lines so printer/HTML can color ins/del lines."""
    lines = difflib.unified_diff(
        original.splitlines(),
        suggested.splitlines(),
        fromfile="original",
        tofile="suggested",
        lineterm="",
    )
    classified = []
    for line in lines:
        if line.startswith(("+++", "---")):
            kind = "header"
        elif line.startswith("@@"):
            kind = "hunk"
        elif line.startswith("+"):
            kind = "ins"
        elif line.startswith("-"):
            kind = "del"
        else:
            kind = "context"
        classified.append({"text": line, "kind": kind})
    return classified


def _format_tags(perf_tags: list[str]) -> list[dict]:
    """Format performance tag slugs for display, mirrors ReportBuilder._format_performance_tags."""
    if not perf_tags or perf_tags == [_NORMAL_TAG]:
        return []
    return [{"name": tag.upper(), "slug": tag} for tag in perf_tags]


def _resume_commands(run_id: str, n_suggestions: int) -> list[str]:
    commands = [
        f"%perfmonitor_ai_review --resume {run_id} --select {index}"
        for index in range(1, n_suggestions + 1)
    ]
    if n_suggestions:
        commands.append(
            f'%perfmonitor_ai_review --resume {run_id} --select 1 '
            f'--note "use multiprocessing instead of joblib"'
        )
    return commands


class AIReviewPrinter:
    """Plain-text rendering of an AI review run, mirrors ReportPrinter."""

    def print(self, state: OptimizationState) -> None:
        run_id = state.get("run_id", "")
        tags = _format_tags(state["perf_tags"])
        tags_line = " | ".join(tag["name"] for tag in tags) or "NORMAL"

        print("-" * 52)
        print(f"[JUmPER AI]  run_id: {run_id}   {tags_line}")
        print("-" * 52)
        print(f"Analysis: {state['analysis']}")
        print()

        for index, suggestion in enumerate(state["suggestions"], start=1):
            print(f"Option {index} — {suggestion.title}")
            print(f"  {suggestion.description}")
            for line in _diff_lines(state["cell_code"], suggestion.code):
                print(f"  {line['text']}")
            print()

        print("To apply:")
        for command in _resume_commands(run_id, len(state["suggestions"])):
            print(f"  {command}")


class AIReviewDisplayer:
    """HTML rendering of an AI review run via Jinja2, mirrors ReportDisplayer."""

    def __init__(self, templates_dir=None):
        self.templates_dir = (
            Path(templates_dir) if templates_dir
            else Path(__file__).parent.parent.parent.parent / "templates"
        )

    def display(self, state: OptimizationState) -> None:
        run_id = state.get("run_id", "")
        tags = _format_tags(state["perf_tags"])
        options = [
            {
                "index": index,
                "title": suggestion.title,
                "description": suggestion.description,
                "diff": _diff_lines(state["cell_code"], suggestion.code),
                "resume_command": f"%perfmonitor_ai_review --resume {run_id} --select {index}",
            }
            for index, suggestion in enumerate(state["suggestions"], start=1)
        ]

        env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=select_autoescape(["html", "xml"]),
        )
        template_path = Path("ai_review") / "ai_review.html"
        template = env.get_template(template_path.as_posix())
        try:
            styles_path = self.templates_dir / "ai_review" / "styles.css"
            inline_styles = styles_path.read_text(encoding="utf-8") if styles_path.exists() else ""
        except Exception:
            inline_styles = ""

        html = template.render(
            run_id=run_id,
            analysis=state["analysis"],
            tags=tags,
            options=options,
            resume_commands=_resume_commands(run_id, len(state["suggestions"])),
            inline_styles=inline_styles,
        )
        display(HTML(html))


class UnavailableAIReviewDisplayer:
    def __init__(self, reason: str = "Display not available."):
        self._reason = reason

    def display(self, state: OptimizationState) -> None:
        logger.info(
            EXTENSION_INFO_MESSAGES[ExtensionInfoCode.HTML_REPORTS_NOT_AVAILABLE].format(
                reason=self._reason
            )
        )


class AIReviewDisplay:
    """Adapter wrapping printer and displayer, mirrors PerformanceReporter."""

    def __init__(self, printer: AIReviewPrinter, displayer):
        self.printer = printer
        self.displayer = displayer

    def print(self, state: OptimizationState) -> None:
        self.printer.print(state)

    def display(self, state: OptimizationState) -> None:
        self.displayer.display(state)


def build_ai_review_display(
    templates_dir=None,
    display_disabled: bool = False,
    display_disabled_reason: str = "Display not available.",
) -> AIReviewDisplay:
    """Build an AIReviewDisplay; mirrors build_performance_reporter()."""
    printer = AIReviewPrinter()
    if display_disabled:
        displayer = UnavailableAIReviewDisplayer(reason=display_disabled_reason)
    else:
        displayer = AIReviewDisplayer(templates_dir)
    return AIReviewDisplay(printer, displayer)
