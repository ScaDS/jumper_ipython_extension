from types import SimpleNamespace
from unittest.mock import Mock

from jumper_extension.adapters.ai_reviewer.agent.graph import build_resume_graph, build_review_graph
from jumper_extension.adapters.ai_reviewer.agent.state import Suggestion, empty_state


def _make_review_llm(analysis="The loop is CPU-bound.", suggestions=None):
    suggestions = suggestions or [
        SimpleNamespace(title="Vectorize", description="Use numpy", code="x = vectorized()", target_cell_index=None),
        SimpleNamespace(title="Cache", description="Memoize results", code="x = cached()", target_cell_index=None),
    ]
    structured_llm = Mock()
    structured_llm.invoke = Mock(return_value=SimpleNamespace(suggestions=suggestions))

    llm = Mock()
    llm.invoke = Mock(return_value=SimpleNamespace(content=analysis))
    llm.with_structured_output = Mock(return_value=structured_llm)
    return llm


def _make_collector(collected):
    collector = Mock()
    collector.collect = Mock(return_value=collected)
    return collector


def test_review_graph_collects_analyzes_suggests_and_displays():
    collected = empty_state(run_id="placeholder", cell_range=(1, 2), level="process")
    collected["cell_code"] = "for i in range(n):\n    work(i)"
    collected["perf_tags"] = ["cpu_bound"]

    llm = _make_review_llm()
    collector = _make_collector(collected)
    review_display = Mock()

    graph = build_review_graph(llm, collector, review_display)
    final_state = graph.invoke(empty_state(run_id="abc123", cell_range=(1, 2)))

    assert final_state["run_id"] == "abc123"
    assert final_state["cell_code"] == "for i in range(n):\n    work(i)"
    assert final_state["analysis"] == "The loop is CPU-bound."
    assert final_state["suggestions"] == [
        Suggestion(title="Vectorize", description="Use numpy", code="x = vectorized()"),
        Suggestion(title="Cache", description="Memoize results", code="x = cached()"),
    ]
    review_display.display.assert_called_once()
    assert review_display.display.call_args[0][0]["run_id"] == "abc123"


def _resume_state(note=""):
    state = empty_state(run_id="abc123")
    state["cell_code"] = "for i in range(n):\n    work(i)"
    state["analysis"] = "The loop is CPU-bound."
    state["suggestions"] = [
        Suggestion(title="Vectorize", description="Use numpy", code="x = vectorized()"),
    ]
    state["chosen_index"] = 0
    state["note"] = note
    return state


def test_resume_graph_applies_suggestion_directly_without_refine():
    llm = Mock()
    shell = Mock()

    graph = build_resume_graph(llm, shell)
    final_state = graph.invoke(_resume_state(note=""))

    llm.invoke.assert_not_called()
    shell.set_next_input.assert_called_once_with("x = vectorized()")
    assert final_state["applied"] is True


def test_resume_graph_refines_then_applies_suggestion():
    llm = Mock()
    llm.invoke = Mock(return_value=SimpleNamespace(content="x = vectorized_parallel()"))
    shell = Mock()

    graph = build_resume_graph(llm, shell)
    final_state = graph.invoke(_resume_state(note="use multiprocessing"))

    llm.invoke.assert_called_once()
    shell.set_next_input.assert_called_once_with("x = vectorized_parallel()")
    assert final_state["refined_code"] == "x = vectorized_parallel()"
    assert final_state["applied"] is True
