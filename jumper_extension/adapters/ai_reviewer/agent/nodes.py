import difflib
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from jumper_extension.adapters.ai_reviewer.agent.prompts import (
    ANALYZE_SYSTEM_PROMPT,
    REFINE_SYSTEM_PROMPT,
    SUGGEST_SYSTEM_PROMPT,
)
from jumper_extension.adapters.ai_reviewer.agent.state import OptimizationState, Suggestion
from jumper_extension.adapters.ai_reviewer.context.collector import ContextCollector
from jumper_extension.adapters.ai_reviewer.ui.review_display import AIReviewDisplay
from jumper_extension.core.messages import EXTENSION_ERROR_MESSAGES, ExtensionErrorCode

logger = logging.getLogger("extension")


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


class _SuggestionListSchema(BaseModel):
    """Structured-output schema for the ranked list of suggestions."""
    suggestions: list[_SuggestionSchema] = Field(
        description="Optimization options, ordered from most to least impactful"
    )


def collect_context_node(state: OptimizationState, collector: ContextCollector) -> OptimizationState:
    """Populate *state* with cell code, performance data, tags and hardware info.

    The incoming ``cell_range``/``level`` describe what to collect; the
    collector resolves ``cell_range`` (e.g. ``None`` -> last non-short cell)
    and returns it alongside the gathered data.
    """
    collected = collector.collect(state["cell_range"], state["level"])
    if collected is None:
        logger.warning(EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_PERFORMANCE_DATA])
        return state

    # collected["run_id"] is a placeholder; keep the run_id assigned by the service
    return {**state, **collected, "run_id": state["run_id"]}


def analyze_bottlenecks_node(state: OptimizationState, llm: BaseChatModel) -> OptimizationState:
    """LLM call #1: produce a short bottleneck narrative for the cell."""
    user_prompt = (
        f"Cell source code:\n{state['cell_code']}\n\n"
        f"Performance tags: {', '.join(state['perf_tags']) or 'none'}\n"
        f"Performance summary (mean/max per metric): {state['perf_summary']}\n"
        f"Hardware: {state['hardware_info']}"
    )
    response = llm.invoke([
        SystemMessage(content=ANALYZE_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])
    return {**state, "analysis": response.content}


def generate_suggestions_node(state: OptimizationState, llm: BaseChatModel) -> OptimizationState:
    """LLM call #2: produce a structured list of optimization suggestions."""
    user_prompt = (
        f"Bottleneck analysis:\n{state['analysis']}\n\n"
        f"Cell source code:\n{state['cell_code']}\n\n"
        f"Hardware: {state['hardware_info']}\n"
        f"Available libraries: {state['env_info']}"
    )
    structured_llm = llm.with_structured_output(_SuggestionListSchema)
    response = structured_llm.invoke([
        SystemMessage(content=SUGGEST_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])
    suggestions = [
        Suggestion(title=item.title, description=item.description, code=item.code)
        for item in response.suggestions
    ]
    return {**state, "suggestions": suggestions}


def display_results_node(state: OptimizationState, review_display: AIReviewDisplay) -> OptimizationState:
    """Render the analysis, options and resume commands (non-blocking)."""
    review_display.display(state)
    return state


def _other_options_as_diffs(
    cell_code: str,
    suggestions: list[Suggestion],
    chosen_index: int,
) -> str:
    parts = []
    for i, s in enumerate(suggestions):
        if i == chosen_index:
            continue
        diff_lines = difflib.unified_diff(
            cell_code.splitlines(),
            s.code.splitlines(),
            fromfile="original",
            tofile=f"option {i + 1}",
            lineterm="",
        )
        parts.append(f"Option {i + 1} — {s.title}:\n" + "\n".join(diff_lines))
    return "\n\n".join(parts)


def refine_suggestion_node(state: OptimizationState, llm: BaseChatModel) -> OptimizationState:
    """LLM call #3: rewrite the chosen suggestion per the custom instruction."""
    chosen = state["suggestions"][state["chosen_index"]]
    other_diffs = _other_options_as_diffs(
        state["cell_code"],
        state["suggestions"],
        state["chosen_index"],
    )
    user_prompt = (
        f"Original bottleneck analysis:\n{state['analysis']}\n\n"
        + (f"Other proposed options (as diffs vs original cell code):\n{other_diffs}\n\n" if other_diffs else "")
        + f"Selected option (Option {state['chosen_index'] + 1} — {chosen.title}) — full code:\n{chosen.code}\n\n"
        f"Custom instruction:\n{state['custom_instruction']}"
    )
    response = llm.invoke([
        SystemMessage(content=REFINE_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])
    return {**state, "refined_code": response.content}


def apply_suggestion_node(state: OptimizationState, shell: Any) -> OptimizationState:
    """Place the (possibly refined) suggestion code into the next cell."""
    suggestion = state["suggestions"][state["chosen_index"]]
    code = state["refined_code"] or suggestion.code
    shell.set_next_input(code)
    return {**state, "applied": True}


def _should_refine(state: OptimizationState) -> str:
    """Conditional-edge router for the resume graph."""
    return "refine" if state["custom_instruction"] else "apply"
