"""Decide what each suggestion is worth, by running it.

Measuring is strictly one variant at a time - anything else would have them
compete for the very resources being measured, and the baseline was recorded
alone. Only the repair calls overlap: those are network waits, and letting one
variant's fix travel while another is being timed costs the measurement nothing.
"""
import logging
from concurrent.futures import Future, ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import Callable

from jumper_extension.adapters.ai_reviewer.benchmark import fingerprint
from jumper_extension.adapters.ai_reviewer.benchmark.models import (
    FAILED,
    OK,
    BenchmarkResult,
    RunOutcome,
)
from jumper_extension.adapters.ai_reviewer.benchmark.runner import BenchmarkRunner, median_of

logger = logging.getLogger("extension")

BASELINE_LABEL = "baseline"


class _Candidate:
    def __init__(self, label: str, code: str, attempts_left: int):
        self.label = label
        self.code = code
        self.attempts_left = attempts_left
        self.attempts = 1
        self.error = ""


class BenchmarkOrchestrator:
    """Times the baseline and every suggestion, repairing what fails."""

    def __init__(
        self,
        runner: BenchmarkRunner,
        fix_fn: Callable[[str, str], str],
        runs: int = 3,
        fix_attempts: int = 3,
        timeout_factor: float = 10.0,
    ):
        self.runner = runner
        self.fix_fn = fix_fn
        self.runs = max(1, runs)
        self.fix_attempts = max(1, fix_attempts)
        self.timeout_factor = timeout_factor
        # label -> the code that was actually measured, once repairs settled
        self.final_code: dict[str, str] = {}

    def run(self, baseline_code: str, variants: list[tuple[str, str]]) -> dict:
        """Benchmark *baseline_code*, then each ``(label, code)`` variant.

        Returns ``{label: BenchmarkResult}``; the baseline is under
        ``BASELINE_LABEL``. An empty dict means the baseline itself would not
        run, which leaves nothing to compare against.
        """
        baseline_runs = self._measure(baseline_code, BASELINE_LABEL, timeout=None)
        if isinstance(baseline_runs, RunOutcome):
            logger.warning(
                "[JUmPER]: the cell under review did not replay cleanly, so there is "
                f"nothing to compare against: {baseline_runs.error}"
            )
            return {}

        duration, metrics = median_of(baseline_runs)
        baseline = BenchmarkResult(
            label=BASELINE_LABEL,
            status=OK,
            duration_s=duration,
            metrics=metrics,
            correctness=fingerprint.MATCH,
        )
        prints = baseline_runs[-1].fingerprints
        timeout = self._timeout_from(baseline_runs, duration)

        results = {BASELINE_LABEL: baseline}
        results.update(self._benchmark_variants(variants, timeout, duration, prints))
        return results

    def _benchmark_variants(
        self,
        variants: list[tuple[str, str]],
        timeout: float,
        baseline_duration: float,
        baseline_prints: dict,
    ) -> dict:
        results: dict = {}
        pending = [_Candidate(label, code, self.fix_attempts) for label, code in variants]
        repairing: dict[Future, _Candidate] = {}

        with ThreadPoolExecutor(max_workers=max(1, len(pending))) as pool:
            # Syntax is checkable for free, so broken code never costs a replay.
            runnable = []
            for candidate in pending:
                if self._syntax_ok(candidate):
                    runnable.append(candidate)
                elif not self._submit_fix(candidate, pool, repairing):
                    results[candidate.label] = _failed(candidate)
            pending = runnable

            while pending or repairing:
                if pending:
                    candidate = pending.pop(0)
                    outcome = self._measure(candidate.code, candidate.label, timeout)
                    if isinstance(outcome, list):
                        self.final_code[candidate.label] = candidate.code
                        results[candidate.label] = self._verdict(
                            candidate, outcome, baseline_duration, baseline_prints
                        )
                    else:
                        candidate.error = outcome.error
                        if not self._submit_fix(candidate, pool, repairing):
                            results[candidate.label] = _failed(candidate)
                    continue

                done, _ = wait(list(repairing), return_when=FIRST_COMPLETED)
                for future in done:
                    candidate = repairing.pop(future)
                    fixed = _fixed_code(future, candidate)
                    if fixed is None:
                        results[candidate.label] = _failed(candidate)
                        continue
                    candidate.code = fixed
                    candidate.attempts += 1
                    pending.append(candidate)
        return results

    def _submit_fix(self, candidate: _Candidate, pool: ThreadPoolExecutor, repairing: dict) -> bool:
        """Queue a repair for *candidate*; False once its attempts are spent."""
        if candidate.attempts_left <= 0:
            return False
        candidate.attempts_left -= 1
        future = pool.submit(self.fix_fn, candidate.code, candidate.error)
        repairing[future] = candidate
        return True

    def _syntax_ok(self, candidate: _Candidate) -> bool:
        try:
            compile(candidate.code, "<suggestion>", "exec")
            return True
        except SyntaxError as error:
            candidate.error = f"{error.__class__.__name__}: {error}"
            return False

    def _measure(self, code: str, label: str, timeout: float | None):
        """Time *code* ``runs`` times, or return the outcome that stopped it."""
        outcomes = []
        for index in range(self.runs):
            outcome = self.runner.run_once(code, tag=f"{label}_{index}", timeout=timeout)
            if not outcome.ok:
                return outcome
            outcomes.append(outcome)
        return outcomes

    def _timeout_from(self, baseline_runs: list[RunOutcome], duration: float) -> float:
        """Budget a variant gets: the prefix it must replay, plus room to be slow."""
        walls = [run.wall_s for run in baseline_runs if run.wall_s]
        prefix = max(walls) - duration if walls else 0.0
        return max(prefix, 0.0) * 2 + max(duration * self.timeout_factor, 5.0)

    def _verdict(
        self,
        candidate: _Candidate,
        outcomes: list[RunOutcome],
        baseline_duration: float,
        baseline_prints: dict,
    ) -> BenchmarkResult:
        duration, metrics = median_of(outcomes)
        correctness, differing = fingerprint.compare_all(
            baseline_prints, outcomes[-1].fingerprints
        )
        return BenchmarkResult(
            label=candidate.label,
            status=OK,
            attempts=candidate.attempts,
            duration_s=duration,
            metrics=metrics,
            speedup=round(baseline_duration / duration, 2) if duration else None,
            correctness=correctness,
            differing_names=differing,
        )


def _fixed_code(future: Future, candidate: _Candidate) -> str | None:
    try:
        fixed = future.result()
    except Exception as error:
        logger.debug("[JUmPER]: repair call failed for %s: %s", candidate.label, error)
        return None
    return fixed or None


def _failed(candidate: _Candidate) -> BenchmarkResult:
    return BenchmarkResult(
        label=candidate.label,
        status=FAILED,
        attempts=candidate.attempts,
        error=candidate.error,
    )
