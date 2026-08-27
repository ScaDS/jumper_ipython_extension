import logging
import sys
from unittest.mock import Mock, patch

from jumper_extension.adapters.ai_reviewer.reviewer import (
    AIReviewer,
    AIReviewerProtocol,
    UnavailableAIReviewer,
    _ai_extras_install_cmd,
    build_ai_reviewer,
)
from jumper_extension.monitor.common import UnavailablePerformanceMonitor


def _make_reporter():
    return Mock()


def test_build_ai_reviewer_returns_real_adapter_when_extras_installed():
    reviewer = build_ai_reviewer(_make_reporter())

    assert isinstance(reviewer, AIReviewer)
    assert isinstance(reviewer, AIReviewerProtocol)


def test_build_ai_reviewer_falls_back_to_stub_when_extras_missing():
    with patch.dict(sys.modules, {"langgraph": None, "langchain_openai": None}):
        reviewer = build_ai_reviewer(_make_reporter())

    assert isinstance(reviewer, UnavailableAIReviewer)
    assert isinstance(reviewer, AIReviewerProtocol)


def test_unavailable_ai_reviewer_logs_install_hint_for_review_and_resume(caplog):
    reviewer = UnavailableAIReviewer()
    caplog.set_level(logging.INFO, logger="extension")

    reviewer.review(shell=Mock())
    reviewer.resume(shell=Mock(), run_id="abc123", select=1)

    assert caplog.text.count(_ai_extras_install_cmd()) == 2


def test_ai_reviewer_starts_unattached_and_binds_monitor_via_attach():
    reviewer = AIReviewer(_make_reporter())

    assert isinstance(reviewer.monitor, UnavailablePerformanceMonitor)
    assert reviewer.monitor.running is False

    monitor = Mock(running=True)
    reviewer.attach(monitor)

    assert reviewer.monitor is monitor


def test_review_warns_and_skips_when_monitor_not_running(caplog):
    reviewer = AIReviewer(_make_reporter())
    reviewer.attach(Mock(running=False))
    caplog.set_level(logging.WARNING, logger="extension")

    reviewer.review(shell=Mock())

    assert "No active performance monitoring session" in caplog.text
    assert reviewer._pending_reviews == {}


def test_review_invokes_graph_once_and_stores_pending_state():
    reviewer = AIReviewer(_make_reporter())
    reviewer.attach(Mock(running=True))

    final_state = {"run_id": "ignored", "suggestions": []}
    fake_graph = Mock()
    fake_graph.invoke = Mock(return_value=final_state)
    reviewer._get_review_graph = Mock(return_value=fake_graph)

    reviewer.review(shell=Mock(), cell_range=(1, 2), level="user")
    reviewer.review(shell=Mock(), cell_range=(1, 2), level="user")

    reviewer._get_review_graph.assert_called()
    initial_state = fake_graph.invoke.call_args_list[0][0][0]
    assert initial_state["cell_range"] == (1, 2)
    assert initial_state["level"] == "user"
    assert len(reviewer._pending_reviews) == 2


def test_resume_warns_when_run_id_unknown(caplog):
    reviewer = AIReviewer(_make_reporter())
    caplog.set_level(logging.WARNING, logger="extension")

    reviewer.resume(shell=Mock(), run_id="missing", select=1)

    assert "No pending AI review found for run_id 'missing'" in caplog.text


def test_resume_warns_when_select_out_of_range(caplog):
    reviewer = AIReviewer(_make_reporter())
    reviewer._pending_reviews["abc123"] = {"suggestions": [Mock()]}
    caplog.set_level(logging.WARNING, logger="extension")

    reviewer.resume(shell=Mock(), run_id="abc123", select=2)

    assert "Invalid suggestion index 2 for run_id 'abc123'" in caplog.text


def test_resume_invokes_graph_with_chosen_index_and_stores_result():
    reviewer = AIReviewer(_make_reporter())
    reviewer._pending_reviews["abc123"] = {
        "suggestions": [Mock(), Mock()],
        "chosen_index": None,
        "note": "",
        "refined_code": None,
    }

    final_state = {"applied": True}
    fake_graph = Mock()
    fake_graph.invoke = Mock(return_value=final_state)
    reviewer._get_resume_graph = Mock(return_value=fake_graph)
    shell = Mock()

    reviewer.resume(shell, run_id="abc123", select=2, note="use multiprocessing")

    reviewer._get_resume_graph.assert_called_once_with(shell)
    resume_state = fake_graph.invoke.call_args[0][0]
    assert resume_state["chosen_index"] == 1
    assert resume_state["note"] == "use multiprocessing"
    assert resume_state["refined_code"] is None
    assert reviewer._pending_reviews["abc123"] == final_state
