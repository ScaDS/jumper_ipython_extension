import difflib
import logging
from pathlib import Path

from IPython.display import HTML, display
from jinja2 import Environment, FileSystemLoader, select_autoescape

from jumper_extension.adapters.ai_reviewer.agent.state import OptimizationState, original_code
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


def _target_label(suggestion) -> str:
    """Name the cell an option rewrites; empty for a single-cell review."""
    if suggestion.target_cell_index is None:
        return ""
    return f" (cell {suggestion.target_cell_index})"


_CORRECTNESS_NOTES = {
    "differs": "but its results differ from the original - treat the speedup as unearned",
    "unverified": "results could not be compared",
}


def verdict_line(state: OptimizationState, index: int) -> str:
    """One-line measured verdict for option *index*, empty when not benchmarked."""
    result = state.get("benchmarks", {}).get(str(index))
    if result is None:
        return ""

    if not result.ok:
        return (
            f"Verdict: failed to run after {result.attempts} attempt(s) - "
            f"{_first_line(result.error)}"
        )

    parts = [f"Verdict: {result.speedup}x faster" if (result.speedup or 0) >= 1
             else f"Verdict: {round(1 / result.speedup, 2)}x slower" if result.speedup
             else "Verdict: measured"]
    parts.append(f"{result.duration_s}s vs {_baseline_duration(state)}s")
    note = _CORRECTNESS_NOTES.get(result.correctness)
    if note:
        names = ", ".join(result.differing_names)
        parts.append(f"{note}{f' ({names})' if names else ''}")
    if result.attempts > 1:
        parts.append(f"repaired after {result.attempts - 1} failed attempt(s)")
    return " - ".join(parts)


def _baseline_duration(state: OptimizationState):
    baseline = state.get("benchmarks", {}).get("baseline")
    return baseline.duration_s if baseline else "?"


def _first_line(text: str) -> str:
    lines = [line for line in (text or "").splitlines() if line.strip()]
    return lines[-1] if lines else "no error reported"


def _reasoning_preview(reasoning: str, limit: int = 140) -> str:
    """First-line preview of the reasoning, shown in the collapsed spoiler summary."""
    snippet = " ".join(reasoning.split())
    return snippet if len(snippet) <= limit else snippet[:limit].rstrip() + "…"


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
            print(f"Option {index} — {suggestion.title}{_target_label(suggestion)}")
            print(f"  {suggestion.description}")
            verdict = verdict_line(state, index)
            if verdict:
                print(f"  {verdict}")
            for line in _diff_lines(original_code(state, suggestion), suggestion.code):
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
                "title": f"{suggestion.title}{_target_label(suggestion)}",
                "description": suggestion.description,
                "verdict": verdict_line(state, index),
                "diff": _diff_lines(original_code(state, suggestion), suggestion.code),
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

        analysis_reasoning = state.get("analysis_reasoning", "")
        html = template.render(
            run_id=run_id,
            analysis=state["analysis"],
            analysis_reasoning=analysis_reasoning,
            reasoning_preview=_reasoning_preview(analysis_reasoning),
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
