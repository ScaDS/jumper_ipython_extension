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
    cell_code: str
    perf_summary: dict
    hardware_info: dict
    perf_tags: list[str]
    env_info: dict
    analysis: str
    suggestions: list[Suggestion]
    chosen_index: int | None
    custom_instruction: str
    refined_code: str | None
    applied: bool


def empty_state(
    run_id: str = "",
    cell_range: tuple[int, int] | None = None,
    level: str = "process",
) -> OptimizationState:
    """Build a fresh, blank :class:`OptimizationState` for a new graph run."""
    return OptimizationState(
        run_id=run_id,
        cell_range=cell_range,
        level=level,
        cell_code="",
        perf_summary={},
        hardware_info={},
        perf_tags=[],
        env_info={},
        analysis="",
        suggestions=[],
        chosen_index=None,
        custom_instruction="",
        refined_code=None,
        applied=False,
    )
