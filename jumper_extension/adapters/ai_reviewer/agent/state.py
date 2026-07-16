from dataclasses import dataclass
from typing import TypedDict


@dataclass
class Suggestion:
    """A single optimization suggestion proposed by the LLM.

    ``code`` rewrites exactly one cell: ``target_cell_index`` names which one
    when the review spans a range, and is None when a single cell was reviewed
    and there is nothing to disambiguate.
    """
    title: str
    description: str
    code: str
    target_cell_index: int | None = None


class OptimizationState(TypedDict):
    """Shared state threaded through the AI review LangGraph workflows."""
    run_id: str
    cell_range: tuple[int, int] | None
    level: str
    overrides: dict
    note: str
    cell_code: str
    cell_sources: dict
    timing_info: dict
    perf_summary: dict
    raw_perf: dict
    hardware_info: dict
    perf_tags: list[str]
    env_info: dict
    analysis: str
    analysis_reasoning: str
    suggestions: list[Suggestion]
    chosen_index: int | None
    refined_code: str | None
    applied: bool


def original_code(state: OptimizationState, suggestion: Suggestion) -> str:
    """The code *suggestion* rewrites: its target cell, or the whole selection.

    Diffs and the applied result must line up with the one cell being rewritten,
    never with the marked-up join of a range - otherwise every other cell in the
    range reads as deleted.
    """
    index = suggestion.target_cell_index
    if index is None:
        return state["cell_code"]
    return state["cell_sources"].get(index, state["cell_code"])


def empty_state(
    run_id: str = "",
    cell_range: tuple[int, int] | None = None,
    level: str = "process",
    overrides: dict | None = None,
    note: str = "",
) -> OptimizationState:
    """Build a fresh, blank :class:`OptimizationState` for a new graph run."""
    return OptimizationState(
        run_id=run_id,
        cell_range=cell_range,
        level=level,
        overrides=overrides or {},
        note=note,
        cell_code="",
        cell_sources={},
        timing_info={},
        perf_summary={},
        raw_perf={},
        hardware_info={},
        perf_tags=[],
        env_info={},
        analysis="",
        analysis_reasoning="",
        suggestions=[],
        chosen_index=None,
        refined_code=None,
        applied=False,
    )
