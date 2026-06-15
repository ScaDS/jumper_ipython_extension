import logging
import uuid
from typing import Any, Optional, Protocol, Tuple, runtime_checkable

from jumper_extension.adapters.reporter import PerformanceReporter
from jumper_extension.core.messages import (
    EXTENSION_ERROR_MESSAGES,
    EXTENSION_INFO_MESSAGES,
    ExtensionErrorCode,
    ExtensionInfoCode,
)
from jumper_extension.monitor.common import MonitorProtocol, UnavailablePerformanceMonitor

logger = logging.getLogger("extension")

_AI_EXTRAS_MISSING_REASON = (
    "optional dependencies (langgraph, langchain-openai) are not installed"
)


@runtime_checkable
class AIReviewerProtocol(Protocol):
    """Structural protocol for the AI review adapter used by the service."""
    def attach(self, monitor: MonitorProtocol) -> None: ...

    def review(
        self,
        shell: Any,
        cell_range: Optional[Tuple[int, int]] = None,
        level: str = "process",
    ) -> None: ...

    def resume(
        self,
        shell: Any,
        run_id: str,
        select: int,
        refine: str = "",
    ) -> None: ...


class AIReviewer:
    """Adapter running the LangGraph-based AI optimization-review workflow.

    Builds (and caches) the fresh-run and resume graphs, generates run
    ids, and keeps track of pending reviews so a follow-up
    ``--resume RUN_ID --select N`` can apply one of the suggestions.
    Mirrors :class:`PerformanceReporter` / :class:`PerformanceVisualizer`:
    constructed with the stable ``reporter`` and bound to the live
    monitor via :meth:`attach`.
    """

    def __init__(self, reporter: PerformanceReporter):
        self.reporter = reporter
        self.monitor: MonitorProtocol = UnavailablePerformanceMonitor(
            reason="Monitor has not been started yet."
        )
        self._pending_reviews: dict[str, dict] = {}
        self._review_graph = None
        self._resume_graph = None

    def attach(self, monitor: MonitorProtocol) -> None:
        """Attach started PerformanceMonitor."""
        self.monitor = monitor

    def _get_review_graph(self):
        """Lazily build (and cache) the fresh-run AI review graph.

        Imported lazily so the optional ``[ai]`` extras (langgraph,
        langchain-openai) are only required when this magic is used.
        """
        if self._review_graph is None:
            from jumper_extension.adapters.ai_reviewer.agent.graph import build_review_graph
            from jumper_extension.adapters.ai_reviewer.context.collector import ContextCollector
            from jumper_extension.adapters.ai_reviewer.llm.client import LLMClientConfig, build_llm
            from jumper_extension.adapters.ai_reviewer.ui.review_display import build_ai_review_display

            llm = build_llm(LLMClientConfig.from_config())
            self._review_graph = build_review_graph(
                llm,
                ContextCollector(self),
                build_ai_review_display(),
            )
        return self._review_graph

    def _get_resume_graph(self, shell: Any):
        """Lazily build (and cache) the resume AI review graph."""
        if self._resume_graph is None:
            from jumper_extension.adapters.ai_reviewer.agent.graph import build_resume_graph
            from jumper_extension.adapters.ai_reviewer.llm.client import LLMClientConfig, build_llm

            llm = build_llm(LLMClientConfig.from_config())
            self._resume_graph = build_resume_graph(llm, shell)
        return self._resume_graph

    def review(
        self,
        shell: Any,
        cell_range: Optional[Tuple[int, int]] = None,
        level: str = "process",
    ) -> None:
        """Run the AI-powered performance review on a fresh cell selection.

        Collects the cell code and performance context, asks the LLM to
        identify the bottleneck and propose optimizations, then displays
        the numbered options together with ``--resume`` commands. The
        resulting state is kept in memory under a short run id so a
        follow-up ``resume`` call can apply one of the suggestions.
        """
        if not self.monitor.running:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_ACTIVE_MONITOR]
            )
            return

        from jumper_extension.adapters.ai_reviewer.agent.state import empty_state

        run_id = uuid.uuid4().hex[:8]
        initial_state = empty_state(
            run_id=run_id,
            cell_range=cell_range,
            level=level,
        )
        final_state = self._get_review_graph().invoke(initial_state)
        self._pending_reviews[run_id] = final_state

    def resume(
        self,
        shell: Any,
        run_id: str,
        select: int,
        refine: str = "",
    ) -> None:
        """Apply a previously suggested optimization, optionally refined.

        Loads the state stored under ``run_id`` by a prior ``review``
        run, marks suggestion ``select`` as chosen and runs the resume
        graph: if ``refine`` is provided, the suggestion is rewritten
        per the custom instruction first; either way the resulting code
        is placed into the next cell via ``shell.set_next_input``.
        """
        state = self._pending_reviews.get(run_id)
        if state is None:
            logger.warning(f"[JUmPER]: No pending AI review found for run_id '{run_id}'")
            return

        if not 1 <= select <= len(state["suggestions"]):
            logger.warning(
                f"[JUmPER]: Invalid suggestion index {select} for run_id '{run_id}' "
                f"(expected 1-{len(state['suggestions'])})"
            )
            return

        resume_state = {
            **state,
            "chosen_index": select - 1,
            "custom_instruction": refine,
            "refined_code": None,
        }
        final_state = self._get_resume_graph(shell).invoke(resume_state)
        self._pending_reviews[run_id] = final_state


class UnavailableAIReviewer:
    """A stub that type-checks against AIReviewerProtocol but only logs
    that the optional ``[ai]`` dependencies are missing.
    """
    def __init__(self, reason: str = _AI_EXTRAS_MISSING_REASON):
        self._reason = reason

    def attach(self, monitor: MonitorProtocol) -> None: ...

    def review(
        self,
        shell: Any,
        cell_range: Optional[Tuple[int, int]] = None,
        level: str = "process",
    ) -> None:
        self._warn()

    def resume(
        self,
        shell: Any,
        run_id: str,
        select: int,
        refine: str = "",
    ) -> None:
        self._warn()

    def _warn(self) -> None:
        logger.info(
            EXTENSION_INFO_MESSAGES[ExtensionInfoCode.AI_REVIEW_NOT_AVAILABLE].format(
                reason=self._reason
            )
        )


def build_ai_reviewer(reporter: PerformanceReporter) -> AIReviewerProtocol:
    """Build an AI reviewer adapter attached to *reporter*.

    Falls back to :class:`UnavailableAIReviewer` when the optional
    ``[ai]`` extras (``langgraph``, ``langchain-openai``) are not
    installed, so ``%perfmonitor_ai_review`` can explain how to enable
    the feature instead of raising an ``ImportError``.
    """
    try:
        import langgraph  # noqa: F401
        import langchain_openai  # noqa: F401
    except ImportError:
        return UnavailableAIReviewer()
    return AIReviewer(reporter)
