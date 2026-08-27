"""Print the exact messages each node sends to the LLM, on a synthetic state.

    python -m jumper_extension.adapters.ai_reviewer.agent.preview [--strategy S] [--note TEXT]

Unlike ``prompts/__main__`` (system half only), this shows the full
``[SystemMessage, HumanMessage]`` for analyze / suggest / refine, built by the
same ``build_*_messages`` functions the real nodes use - so there is no drift
between what is printed and what the LLM actually receives. The state is
synthetic; a strategy that disables a context source empties the matching
field here too, mirroring the collector.
"""
import argparse

from langchain_core.messages import BaseMessage

from jumper_extension.adapters.ai_reviewer.agent.nodes import (
    build_analyze_messages,
    build_refine_messages,
    build_suggest_messages,
)
from jumper_extension.adapters.ai_reviewer.agent.state import OptimizationState, Suggestion, empty_state
from jumper_extension.adapters.ai_reviewer.context.collector import _SOURCE_FIELDS
from jumper_extension.adapters.ai_reviewer.strategy import get_strategy, strategy_ids


def _synthetic_state(strategy: str, note: str) -> OptimizationState:
    overrides = get_strategy(strategy).overrides
    state = empty_state(run_id="preview", cell_range=(1, 1), overrides=overrides, note=note)
    state["cell_code"] = (
        "import numpy as np\n"
        "result = sum(np.sqrt(i) for i in range(10_000_000))"
    )
    state["timing_info"] = {
        "total_duration_s": 2.13,
        "per_cell_duration_s": {1: 2.13},
    }
    state["perf_tags"] = ["cpu_bound"]
    state["perf_summary"] = {
        "overall": {
            "cpu": {"mean": 82.0, "max": 99.0},
            "memory": {"mean": 1.2, "max": 1.5},
        },
    }
    state["raw_perf"] = {
        "time": [0.0, 0.5, 1.0, 1.5, 2.0],
        "cell_index": [1, 1, 1, 1, 1],
        "cpu": [14.0, 71.0, 92.0, 99.0, 88.0],
        "memory": [1.0, 1.1, 1.2, 1.2, 1.2],
    }
    state["hardware_info"] = {
        "num_cpus": 8,
        "num_gpus": 1,
        "gpu_name": "NVIDIA A100",
        "memory_limits": {"process": 32.0},
    }
    state["env_info"] = {"numpy": "1.26.0", "torch": "2.3.0", "cupy": "13.0.0"}
    state["analysis"] = (
        "The cell is CPU-bound: a serial Python generator over 10M sqrt calls "
        "saturates a single core while the GPU sits idle."
    )
    state["suggestions"] = [
        Suggestion(
            title="Vectorize with numpy",
            description="Replace the Python loop with a vectorized numpy reduction.",
            code="import numpy as np\nresult = np.sqrt(np.arange(10_000_000)).sum()",
        ),
        Suggestion(
            title="Offload to GPU with cupy",
            description="Move the reduction onto the idle GPU via cupy.",
            code="import cupy as cp\nresult = float(cp.sqrt(cp.arange(10_000_000)).sum())",
        ),
    ]
    state["chosen_index"] = 0

    # Mirror the collector: honor each source's default, empty the disabled ones.
    for source_id, (field, empty, default) in _SOURCE_FIELDS.items():
        if overrides.get(source_id, default) is False:
            state[field] = empty
    return state


def _print_messages(title: str, messages: list[BaseMessage]) -> None:
    print("=" * 72)
    print(f"# {title}")
    print("=" * 72)
    for message in messages:
        print(f"----- {message.__class__.__name__} -----")
        print(message.content)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the full LLM input per node on a synthetic state.")
    parser.add_argument("--strategy", default="faster", choices=strategy_ids())
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    state = _synthetic_state(args.strategy, args.note)
    label = f"strategy={args.strategy}, note={args.note!r}"
    _print_messages(f"analyze   ({label})", build_analyze_messages(state))
    _print_messages(f"suggest   ({label})", build_suggest_messages(state))
    _print_messages(f"refine    ({label})", build_refine_messages(state))


if __name__ == "__main__":
    main()
