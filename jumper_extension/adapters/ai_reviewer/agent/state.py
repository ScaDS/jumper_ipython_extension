from dataclasses import dataclass
from typing import TypedDict


@dataclass
class Suggestion:
    """A single optimization suggestion proposed by the LLM."""
    title: str
    description: str
    code: str


class OptimizationState(TypedDict):
    """Shared state threaded through the AI review LangGraph workflows."""
    run_id: str
    cell_range: tuple[int, int] | None
    level: str
    overrides: dict
    note: str
    cell_code: str
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
