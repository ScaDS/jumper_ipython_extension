from types import SimpleNamespace
from unittest.mock import Mock

from jumper_extension.adapters.ai_reviewer.agent.nodes import (
    _other_options_as_diffs,
    _should_refine,
    analyze_bottlenecks_node,
    apply_suggestion_node,
    collect_context_node,
    display_results_node,
    generate_suggestions_node,
    refine_suggestion_node,
)
from jumper_extension.adapters.ai_reviewer.agent.state import Suggestion, empty_state


def _state_with_suggestions(**overrides):
    state = empty_state(run_id="abc123")
    state["cell_code"] = "for i in range(n):\n    work(i)"
    state["analysis"] = "The loop is CPU-bound and runs serially."
    state["suggestions"] = [
        Suggestion(title="Vectorize loop", description="Use numpy", code="work_vectorized(n)"),
        Suggestion(title="Parallelize", description="Use joblib", code="parallel_work(n)"),
    ]
    state.update(overrides)
    return state


def test_collect_context_node_merges_collected_data_and_keeps_run_id():
    state = empty_state(run_id="abc123", cell_range=(1, 2), level="user")
    collected = empty_state(run_id="placeholder", cell_range=(1, 2), level="user")
    collected["cell_code"] = "x = 1"
    collector = Mock()
    collector.collect = Mock(return_value=collected)

    result = collect_context_node(state, collector)

    collector.collect.assert_called_once_with((1, 2), "user", {})
    assert result["run_id"] == "abc123"
    assert result["cell_code"] == "x = 1"


def test_collect_context_node_returns_state_unchanged_when_no_data():
    state = empty_state(run_id="abc123")
    collector = Mock()
    collector.collect = Mock(return_value=None)

    result = collect_context_node(state, collector)

    assert result == state


def test_analyze_bottlenecks_node_sets_analysis_from_llm_response():
    state = empty_state(run_id="abc123")
    state["cell_code"] = "x = compute()"
    state["perf_tags"] = ["cpu_bound"]
    state["perf_summary"] = {"cpu": {"mean": 80.0, "max": 99.0}}
    state["hardware_info"] = {"num_cpus": 8}

    llm = Mock()
    llm.invoke = Mock(return_value=SimpleNamespace(content="CPU is the bottleneck."))

    result = analyze_bottlenecks_node(state, llm)

    assert result["analysis"] == "CPU is the bottleneck."
    llm.invoke.assert_called_once()
    messages = llm.invoke.call_args[0][0]
    assert "x = compute()" in messages[1].content
    assert "cpu_bound" in messages[1].content


def test_generate_suggestions_node_builds_suggestions_from_structured_output():
    state = empty_state(run_id="abc123")
    state["cell_code"] = "x = compute()"
    state["analysis"] = "CPU is the bottleneck."
    state["env_info"] = {"numpy": "1.26.0", "torch": "2.3.0"}

    structured_response = SimpleNamespace(suggestions=[
        SimpleNamespace(title="Vectorize", description="Use numpy", code="x = vectorized()"),
        SimpleNamespace(title="Cache", description="Memoize results", code="x = cached()"),
    ])
    structured_llm = Mock()
    structured_llm.invoke = Mock(return_value=structured_response)
    llm = Mock()
    llm.with_structured_output = Mock(return_value=structured_llm)

    result = generate_suggestions_node(state, llm)

    assert result["suggestions"] == [
        Suggestion(title="Vectorize", description="Use numpy", code="x = vectorized()"),
        Suggestion(title="Cache", description="Memoize results", code="x = cached()"),
    ]
    prompt = structured_llm.invoke.call_args[0][0][1].content
    assert "Available libraries" in prompt
    assert "numpy" in prompt
    assert "torch" in prompt


def test_display_results_node_delegates_to_review_display_and_keeps_state():
    state = _state_with_suggestions()
    review_display = Mock()

    result = display_results_node(state, review_display)

    review_display.display.assert_called_once_with(state)
    assert result == state


def test_refine_suggestion_node_sets_refined_code_from_llm_response():
    state = _state_with_suggestions(chosen_index=0, note="use multiprocessing")

    llm = Mock()
    llm.invoke = Mock(return_value=SimpleNamespace(content="work_multiprocessed(n)"))

    result = refine_suggestion_node(state, llm)

    assert result["refined_code"] == "work_multiprocessed(n)"
    prompt = llm.invoke.call_args[0][0][1].content
    assert "work_vectorized(n)" in prompt        # chosen full code
    assert "use multiprocessing" in prompt       # custom instruction
    assert "Option 2" in prompt                  # other option present as diff
    assert "Parallelize" in prompt


def test_other_options_as_diffs_skips_chosen_and_returns_unified_diff():
    suggestions = [
        Suggestion(title="Vectorize", description="", code="x = np.array(data)"),
        Suggestion(title="Cache", description="", code="x = cached(data)"),
    ]

    result = _other_options_as_diffs(
        cell_code="x = list(data)",
        suggestions=suggestions,
        chosen_index=0,
    )

    assert "Option 1" not in result
    assert "Option 2 — Cache" in result
    assert "-x = list(data)" in result
    assert "+x = cached(data)" in result


def test_other_options_as_diffs_returns_empty_string_for_single_suggestion():
    suggestions = [Suggestion(title="Only", description="", code="x = 1")]

    result = _other_options_as_diffs("x = 0", suggestions, chosen_index=0)

    assert result == ""


def test_apply_suggestion_node_sets_next_input_to_chosen_suggestion_when_not_refined():
    state = _state_with_suggestions(chosen_index=1, refined_code=None)
    shell = Mock()

    result = apply_suggestion_node(state, shell)

    shell.set_next_input.assert_called_once_with("parallel_work(n)")
    assert result["applied"] is True


def test_apply_suggestion_node_prefers_refined_code_when_present():
    state = _state_with_suggestions(chosen_index=0, refined_code="work_refined(n)")
    shell = Mock()

    apply_suggestion_node(state, shell)

    shell.set_next_input.assert_called_once_with("work_refined(n)")


def test_should_refine_routes_based_on_note():
    assert _should_refine(_state_with_suggestions(note="use multiprocessing")) == "refine"
    assert _should_refine(_state_with_suggestions(note="")) == "apply"
