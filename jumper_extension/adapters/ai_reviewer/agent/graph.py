from functools import partial
from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from jumper_extension.adapters.ai_reviewer.agent.nodes import (
    _should_refine,
    analyze_bottlenecks_node,
    apply_suggestion_node,
    collect_context_node,
    display_results_node,
    generate_suggestions_node,
    refine_suggestion_node,
)
from jumper_extension.adapters.ai_reviewer.agent.state import OptimizationState
from jumper_extension.adapters.ai_reviewer.context.collector import ContextCollector
from jumper_extension.adapters.ai_reviewer.ui.review_display import AIReviewDisplay


def build_review_graph(
    llm: BaseChatModel,
    collector: ContextCollector,
    review_display: AIReviewDisplay,
) -> Any:
    """Build the fresh-run graph: collect -> analyze -> suggest -> display -> END."""
    graph = StateGraph(OptimizationState)
    graph.add_node("collect_context", partial(collect_context_node, collector=collector))
    graph.add_node("analyze_bottlenecks", partial(analyze_bottlenecks_node, llm=llm))
    graph.add_node("generate_suggestions", partial(generate_suggestions_node, llm=llm))
    graph.add_node("display_results", partial(display_results_node, review_display=review_display))

    graph.add_edge(START, "collect_context")
    graph.add_edge("collect_context", "analyze_bottlenecks")
    graph.add_edge("analyze_bottlenecks", "generate_suggestions")
    graph.add_edge("generate_suggestions", "display_results")
    graph.add_edge("display_results", END)

    return graph.compile()


def build_resume_graph(llm: BaseChatModel, shell: Any) -> Any:
    """Build the resume graph: optionally refine, then apply the suggestion."""
    graph = StateGraph(OptimizationState)
    graph.add_node("refine_suggestion", partial(refine_suggestion_node, llm=llm))
    graph.add_node("apply_suggestion", partial(apply_suggestion_node, shell=shell))

    graph.add_conditional_edges(
        START,
        _should_refine,
        {"refine": "refine_suggestion", "apply": "apply_suggestion"},
    )
    graph.add_edge("refine_suggestion", "apply_suggestion")
    graph.add_edge("apply_suggestion", END)

    return graph.compile()


if __name__ == "__main__":
    review_graph = build_review_graph(llm=None, collector=None, review_display=None)
    resume_graph = build_resume_graph(llm=None, shell=None)

    print("=== Review Graph ===")
    print(review_graph.get_graph().draw_ascii())
    print()
    print("=== Resume Graph ===")
    print(resume_graph.get_graph().draw_ascii())
