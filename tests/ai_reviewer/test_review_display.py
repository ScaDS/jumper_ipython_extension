from unittest.mock import patch

from jumper_extension.adapters.ai_reviewer.agent.state import Suggestion, empty_state
from jumper_extension.adapters.ai_reviewer.benchmark.models import FAILED, OK, BenchmarkResult
from jumper_extension.adapters.ai_reviewer.ui.review_display import (
    AIReviewDisplayer,
    verdict_line,
    verdict_of,
)


def _state(**benchmarks):
    state = empty_state(run_id="abc123")
    state["cell_code"] = "y = slow(x)"
    state["cell_sources"] = {2: "y = slow(x)"}
    state["suggestions"] = [Suggestion("Vectorize", "Use numpy", "y = fast(x)")]
    if benchmarks:
        state["benchmarks"] = {
            "baseline": BenchmarkResult(label="baseline", status=OK, duration_s=4.0),
            **benchmarks,
        }
    return state


def test_no_verdict_when_the_review_was_not_benchmarked():
    assert verdict_of(_state(), 1) is None
    assert verdict_line(_state(), 1) == ""


def test_a_faster_matching_variant_reads_as_good():
    state = _state(**{
        "1": BenchmarkResult(label="1", status=OK, duration_s=1.0, speedup=4.0, correctness="match")
    })

    verdict = verdict_of(state, 1)

    assert verdict["headline"] == "4.0x faster"
    assert verdict["tone"] == "good"
    assert verdict["detail"] == "1.0s vs 4.0s"
    assert verdict["notes"] == []


def test_a_faster_but_diverging_variant_reads_as_bad_not_as_a_win():
    state = _state(**{
        "1": BenchmarkResult(
            label="1",
            status=OK,
            duration_s=0.001,
            speedup=4000.0,
            correctness="differs",
            differing_names=["y"],
        )
    })

    verdict = verdict_of(state, 1)

    assert verdict["tone"] == "bad"
    assert "unearned" in verdict["notes"][0]
    assert "(y)" in verdict["notes"][0]


def test_a_slower_variant_is_reported_as_slower():
    state = _state(**{
        "1": BenchmarkResult(label="1", status=OK, duration_s=8.0, speedup=0.5, correctness="match")
    })

    verdict = verdict_of(state, 1)

    assert verdict["headline"] == "2.0x slower"
    assert verdict["tone"] == "warn"


def test_an_unverifiable_result_is_flagged_without_calling_it_wrong():
    state = _state(**{
        "1": BenchmarkResult(
            label="1", status=OK, duration_s=1.0, speedup=4.0, correctness="unverified"
        )
    })

    verdict = verdict_of(state, 1)

    assert verdict["tone"] == "warn"
    assert verdict["notes"] == ["results could not be compared"]


def test_a_repaired_variant_says_so():
    state = _state(**{
        "1": BenchmarkResult(
            label="1", status=OK, duration_s=1.0, speedup=4.0, correctness="match", attempts=3
        )
    })

    assert verdict_of(state, 1)["notes"] == ["repaired after 2 failed attempt(s)"]


def test_a_failed_variant_reports_the_error_rather_than_a_speedup():
    state = _state(**{
        "1": BenchmarkResult(
            label="1",
            status=FAILED,
            attempts=4,
            error="Traceback...\nAttributeError: no attribute 'nope'",
        )
    })

    verdict = verdict_of(state, 1)

    assert verdict["headline"] == "Failed after 4 attempt(s)"
    assert verdict["tone"] == "bad"
    assert verdict["detail"] == "AttributeError: no attribute 'nope'"


def test_verdict_reaches_the_html_card():
    state = _state(**{
        "1": BenchmarkResult(label="1", status=OK, duration_s=1.0, speedup=4.0, correctness="match")
    })

    with patch("jumper_extension.adapters.ai_reviewer.ui.review_display.display") as displayed:
        AIReviewDisplayer().display(state)

    html = displayed.call_args[0][0].data
    assert "ai-verdict--good" in html
    assert "4.0x faster" in html
