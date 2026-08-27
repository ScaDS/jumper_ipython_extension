"""Live progress for a benchmark, which otherwise goes silent for minutes.

The estimate starts from the baseline and then follows the variants: the
baseline is the slow code being replaced, so it badly over-predicts an
optimized variant, while variants of the same cell resemble each other.
"""
import logging
import statistics
import time

logger = logging.getLogger("extension")


def humanize(seconds: float) -> str:
    """Round a duration to something worth reading in a log line."""
    if seconds < 60:
        return f"{round(seconds)}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{round(minutes)} min"
    hours, rest = divmod(round(minutes), 60)
    return f"{hours}h {rest:02d}m"


class BenchmarkProgress:
    """Reports what the benchmark is doing, and how much of it is left."""

    def __init__(self, total_variants: int, runs: int):
        self.total_variants = total_variants
        self.runs = runs
        self.measured = 0
        self.failed = 0
        self._started = time.perf_counter()
        self._baseline_wall: float | None = None
        self._variant_walls: list[float] = []

    def baseline_run(self, index: int) -> None:
        logger.info(
            f"[JUmPER]: benchmark: measuring the original cell "
            f"(run {index + 1}/{self.runs})"
        )

    def baseline_done(self, duration: float, walls: list[float]) -> None:
        self._baseline_wall = statistics.median(walls) if walls else None
        logger.info(
            f"[JUmPER]: benchmark: the original cell takes {duration}s"
            f"{self._remaining(self.total_variants, prefix='; ', suffix=' to go')}"
        )

    def variant_run(self, position: int, index: int) -> None:
        logger.info(
            f"[JUmPER]: benchmark: option {position}/{self.total_variants} "
            f"(run {index + 1}/{self.runs})"
        )

    def variant_done(self, position: int, summary: str, walls: list[float], outstanding: int) -> None:
        self._variant_walls.extend(walls)
        self.measured += 1
        logger.info(
            f"[JUmPER]: benchmark: option {position}/{self.total_variants} - {summary}"
            f"{self._remaining(outstanding, prefix='; ', suffix=' left')}"
        )

    def variant_validated(self, position: int, attempts: int) -> None:
        """A suggestion parsed (syntax-only mode: no timing to report)."""
        repaired = f" (repaired {attempts - 1}x)" if attempts > 1 else ""
        logger.info(
            f"[JUmPER]: benchmark: option {position}/{self.total_variants} "
            f"syntax valid{repaired}"
        )

    def variant_failed(self, position: int, error: str, attempt: int, attempts: int) -> None:
        logger.info(
            f"[JUmPER]: benchmark: option {position}/{self.total_variants} failed "
            f"({_first_line(error)}); repairing ({attempt}/{attempts})"
        )

    def variant_diverged(
        self,
        position: int,
        names: list[str],
        attempt: int,
        attempts: int,
    ) -> None:
        changed = f" ({', '.join(names)})" if names else ""
        logger.info(
            f"[JUmPER]: benchmark: option {position}/{self.total_variants} ran but its "
            f"results differ{changed}; repairing ({attempt}/{attempts})"
        )

    def variant_gave_up(self, position: int, attempts: int) -> None:
        self.failed += 1
        logger.info(
            f"[JUmPER]: benchmark: option {position}/{self.total_variants} "
            f"gave up after {attempts} repair(s)"
        )

    def finished(self) -> None:
        elapsed = humanize(time.perf_counter() - self._started)
        logger.info(
            f"[JUmPER]: benchmark: finished in {elapsed} - "
            f"{self.measured} measured, {self.failed} failed"
        )

    def _remaining(self, outstanding: int, prefix: str = "", suffix: str = "") -> str:
        """Best current guess at the time left, or nothing if we cannot tell.

        Deliberately vague: it ignores repair round-trips, and until a variant
        has landed it can only assume they cost what the original does.
        """
        per_run = self._per_run_wall()
        if not per_run or outstanding <= 0:
            return ""
        estimate = humanize(per_run * self.runs * outstanding)
        return f"{prefix}~{estimate}{suffix}"

    def _per_run_wall(self) -> float | None:
        if self._variant_walls:
            return statistics.median(self._variant_walls)
        return self._baseline_wall


def _first_line(text: str) -> str:
    lines = [line for line in (text or "").splitlines() if line.strip()]
    return lines[-1][:120] if lines else "no error reported"
