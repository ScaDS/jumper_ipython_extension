import difflib
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from jumper_extension.adapters.ai_reviewer.agent.state import (
    OptimizationState,
    Suggestion,
    original_code,
)
from jumper_extension.adapters.ai_reviewer.context.collector import ContextCollector
from jumper_extension.adapters.ai_reviewer.llm.reasoning import split_reasoning
from jumper_extension.adapters.ai_reviewer.prompts import PromptLibrary
from jumper_extension.adapters.ai_reviewer.ui.review_display import AIReviewDisplay
from jumper_extension.core.messages import EXTENSION_ERROR_MESSAGES, ExtensionErrorCode

logger = logging.getLogger("extension")

# Silent until the "extension" logger is raised to DEBUG; see logging_config.
prompt_logger = logging.getLogger("extension.ai_prompts")

_prompts = PromptLibrary.load()


def _log_request(run_id: str, step: str, messages: list[BaseMessage]) -> None:
    """Record the exact messages *step* sends to the LLM."""
    if not prompt_logger.isEnabledFor(logging.DEBUG):
        return
    for message in messages:
        prompt_logger.debug(
            "run %s | %s | %s:\n%s",
            run_id,
            step,
            message.__class__.__name__,
            message.content,
        )


def _log_reply(run_id: str, step: str, reply) -> None:
    """Record what the LLM replied to *step*, verbatim."""
    prompt_logger.debug("run %s | %s | reply:\n%s", run_id, step, reply)


class _SuggestionSchema(BaseModel):
    """Structured-output schema for a single optimization suggestion."""
    title: str = Field(description="Short (3-6 word) name of the optimization technique")
    description: str = Field(description="One or two sentences explaining the change and its benefit")
    code: str = Field(
        description=(
            "Complete rewritten cell source code implementing the suggestion, "
            "as properly formatted multi-line Python (real newlines between "
            "statements, PEP 8 style) - never a semicolon-joined one-liner"
        )
    )
    target_cell_index: int | None = Field(
        default=None,
        description=(
            "Index of the single cell this rewrite replaces, matching one of the "
            "'# --- cell N ---' markers in the source. Null when only one cell is "
            "under review"
        ),
    )


class _SuggestionListSchema(BaseModel):
    """Structured-output schema for the ranked list of suggestions."""
    suggestions: list[_SuggestionSchema] = Field(
        description="Optimization options, ordered from most to least impactful"
    )


def collect_context_node(state: OptimizationState, collector: ContextCollector) -> OptimizationState:
    """Populate *state* with cell code, performance data, tags and hardware info.

    The strategy's ``overrides`` decide which context sources are gathered;
    disabled sources come back empty.
    """
    collected = collector.collect(state["cell_range"], state["level"], state["overrides"])
    if collected is None:
        logger.warning(EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_PERFORMANCE_DATA])
        return state

    # collected carries only context data; keep the run identity and the
    # strategy inputs that drive the rest of the graph.
    return {
        **state,
        **collected,
        "run_id": state["run_id"],
        "overrides": state["overrides"],
        "note": state["note"],
    }


def _format_timing(timing: dict) -> str:
    """Render durations so each one stays tied to the cell it was measured on."""
    per_cell = ", ".join(
        f"cell {index}: {duration}s"
        for index, duration in timing["per_cell_duration_s"].items()
    )
    return (
        "Execution time, wall-clock including interpreter and magic overhead - "
        f"total: {timing['total_duration_s']}s; per cell - {per_cell}"
    )


def _format_perf_summary(perf_summary: dict) -> str:
    """Render metrics so a range's per-cell breakdown stays readable."""
    lines = [f"Performance summary (mean/max per metric) - overall: {perf_summary['overall']}"]
    per_cell = perf_summary.get("per_cell")
    if per_cell:
        lines.append("Per cell:")
        lines.extend(
            f"  cell {index}: {metrics}"
            for index, metrics in per_cell.items()
        )
    return "\n".join(lines)


def build_analyze_messages(state: OptimizationState) -> list[BaseMessage]:
    """Exact ``[system, human]`` messages sent to the LLM for the analyze step."""
    lines = []
    if state["cell_code"]:
        lines.append(f"Cell source code:\n{state['cell_code']}")
    if state["timing_info"]:
        lines.append(_format_timing(state["timing_info"]))
    if state["perf_tags"]:
        lines.append(f"Performance tags: {', '.join(state['perf_tags'])}")
    if state["perf_summary"]:
        lines.append(_format_perf_summary(state["perf_summary"]))
    if state["raw_perf"]:
        lines.append(f"Raw metric arrays behind the plots (per timestep): {state['raw_perf']}")
    if state["hardware_info"]:
        lines.append(f"Hardware: {state['hardware_info']}")
    return [
        SystemMessage(content=_prompts.render("analyze", state["overrides"], state["note"])),
        HumanMessage(content="\n\n".join(lines)),
    ]


def analyze_bottlenecks_node(state: OptimizationState, llm: BaseChatModel) -> OptimizationState:
    """LLM call #1: produce a short bottleneck narrative for the cell."""
    messages = build_analyze_messages(state)
    _log_request(state["run_id"], "analyze", messages)
    response = llm.invoke(messages)
    _log_reply(state["run_id"], "analyze", response.content)
    analysis, reasoning = split_reasoning(response.content)
    return {**state, "analysis": analysis, "analysis_reasoning": reasoning}


def build_suggest_messages(state: OptimizationState) -> list[BaseMessage]:
    """Exact ``[system, human]`` messages sent to the LLM for the suggest step."""
    lines = [f"Bottleneck analysis:\n{state['analysis']}"]
    if state["cell_code"]:
        lines.append(f"Cell source code:\n{state['cell_code']}")
    if state["hardware_info"]:
        lines.append(f"Hardware: {state['hardware_info']}")
    if state["env_info"]:
        lines.append(f"Available libraries: {state['env_info']}")
    return [
        SystemMessage(content=_prompts.render("suggest", state["overrides"], state["note"])),
        HumanMessage(content="\n\n".join(lines)),
    ]


def generate_suggestions_node(state: OptimizationState, llm: BaseChatModel) -> OptimizationState:
    """LLM call #2: produce a structured list of optimization suggestions."""
    structured_llm = llm.with_structured_output(_SuggestionListSchema)
    messages = build_suggest_messages(state)
    _log_request(state["run_id"], "suggest", messages)
    response = structured_llm.invoke(messages)
    _log_reply(state["run_id"], "suggest", response)
    suggestions = [
        Suggestion(
            title=item.title,
            description=item.description,
            code=item.code,
            target_cell_index=item.target_cell_index,
        )
        for item in response.suggestions
    ]
    return {**state, "suggestions": suggestions}


def display_results_node(state: OptimizationState, review_display: AIReviewDisplay) -> OptimizationState:
    """Render the analysis, options and resume commands (non-blocking)."""
    review_display.display(state)
    return state


def _other_options_as_diffs(
    state: OptimizationState,
    suggestions: list[Suggestion],
    chosen_index: int,
) -> str:
    parts = []
    for i, s in enumerate(suggestions):
        if i == chosen_index:
            continue
        diff_lines = difflib.unified_diff(
            original_code(state, s).splitlines(),
            s.code.splitlines(),
            fromfile="original",
            tofile=f"option {i + 1}",
            lineterm="",
        )
        parts.append(f"Option {i + 1} — {s.title}:\n" + "\n".join(diff_lines))
    return "\n\n".join(parts)


def build_refine_messages(state: OptimizationState) -> list[BaseMessage]:
    """Exact ``[system, human]`` messages sent to the LLM for the refine step."""
    chosen = state["suggestions"][state["chosen_index"]]
    other_diffs = _other_options_as_diffs(
        state,
        state["suggestions"],
        state["chosen_index"],
    )
    user_prompt = (
        f"Original bottleneck analysis:\n{state['analysis']}\n\n"
        + (f"Other proposed options (as diffs vs original cell code):\n{other_diffs}\n\n" if other_diffs else "")
        + f"Selected option (Option {state['chosen_index'] + 1} — {chosen.title}) — full code:\n{chosen.code}\n\n"
        f"Custom instruction:\n{state['note']}"
    )
    return [
        SystemMessage(content=_prompts.render("refine", state["overrides"], state["note"])),
        HumanMessage(content=user_prompt),
    ]


def refine_suggestion_node(state: OptimizationState, llm: BaseChatModel) -> OptimizationState:
    """LLM call #3: rewrite the chosen suggestion per the ``--note`` instruction."""
    messages = build_refine_messages(state)
    _log_request(state["run_id"], "refine", messages)
    response = llm.invoke(messages)
    _log_reply(state["run_id"], "refine", response.content)
    refined_code, _ = split_reasoning(response.content)
    return {**state, "refined_code": refined_code}


def apply_suggestion_node(state: OptimizationState, shell: Any) -> OptimizationState:
    """Place the (possibly refined) suggestion code into the next cell."""
    suggestion = state["suggestions"][state["chosen_index"]]
    code = state["refined_code"] or suggestion.code
    shell.set_next_input(code)
    return {**state, "applied": True}


def _should_refine(state: OptimizationState) -> str:
    """Conditional-edge router for the resume graph."""
    return "refine" if state["note"] else "apply"
