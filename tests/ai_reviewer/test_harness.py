"""Tests for the language-agnostic benchmark replay harness (Design B spine)."""
import json
import os
import sys
import textwrap
import zipfile

import pandas as pd
import pytest

from jumper_extension.adapters.ai_reviewer.benchmark import harness


def _read_cell_history(session_zip: str) -> pd.DataFrame:
    with zipfile.ZipFile(session_zip) as archive:
        with archive.open("cell_history.csv") as handle:
            return pd.read_csv(handle)


def testsynthesize_history_places_target_at_prefix_index():
    df = harness.synthesize_history(
        prefix_count=3,
        target_code="y <- 1",
        language="r",
        start_epoch=1000.0,
        end_epoch=1002.5,
        offset=50.0,
    )
    # One row per prefix cell plus the target, target last and at its index.
    assert len(df) == 4
    target = df.iloc[3]
    assert target["cell_index"] == 3
    assert target["raw_cell"] == "y <- 1"
    assert target["duration"] == pytest.approx(2.5)
    # The child speaks epoch seconds; the row's window is mapped onto the
    # sampler clock by adding the measured offset.
    assert target["start_time"] == pytest.approx(1050.0)
    assert target["end_time"] == pytest.approx(1052.5)
    assert target["wallclock_start_time"] == pytest.approx(1000.0)


def testsynthesize_history_prefix_rows_are_inert_placeholders():
    df = harness.synthesize_history(
        prefix_count=2,
        target_code="z <- 2",
        language="r",
        start_epoch=10.0,
        end_epoch=11.0,
        offset=0.0,
    )
    prefix = df.iloc[:2]
    assert (prefix["duration"] == 0.0).all()
    assert (prefix["cell_index"] == [0, 1]).all()


def testclock_offset_maps_epoch_to_perf_counter():
    # The offset added to an epoch second should land near the perf_counter now.
    offset = harness.clock_offset()
    import time

    mapped = time.time() + offset
    assert abs(mapped - time.perf_counter()) < 0.5


_BUSY_CHILD = textwrap.dedent(
    """
    import json, sys, time
    markers = sys.argv[1]
    start = time.time()
    end_at = time.perf_counter() + float(sys.argv[2])
    while time.perf_counter() < end_at:
        pass
    end = time.time()
    with open(markers, "w") as handle:
        json.dump({"start": start, "end": end}, handle)
    """
)

_FAILING_CHILD = "import sys\nsys.exit(3)\n"


def _write(path: str, text: str) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def test_run_harness_profiles_child_and_exports_session(tmp_path):
    child = _write(str(tmp_path / "child.py"), _BUSY_CHILD)
    markers = str(tmp_path / "markers.json")
    session = str(tmp_path / "session.zip")

    code = harness.run_harness(
        run_cmd=[sys.executable, child, markers, "0.3"],
        session_path=session,
        markers_path=markers,
        interval=0.05,
        prefix_count=2,
        target_code="x <- 1",
        language="r",
        work_dir=str(tmp_path),
    )

    assert code == 0
    assert os.path.exists(session)
    saved = json.load(open(markers))
    history = _read_cell_history(session)
    assert len(history) == 3
    target = history.iloc[2]
    assert target["cell_index"] == 2
    assert target["duration"] == pytest.approx(saved["end"] - saved["start"], abs=0.05)


def test_run_harness_returns_child_exit_code_and_skips_export(tmp_path):
    child = _write(str(tmp_path / "boom.py"), _FAILING_CHILD)
    markers = str(tmp_path / "markers.json")
    session = str(tmp_path / "session.zip")

    code = harness.run_harness(
        run_cmd=[sys.executable, child, markers],
        session_path=session,
        markers_path=markers,
        interval=0.05,
        prefix_count=0,
        target_code="x <- 1",
        language="r",
        work_dir=str(tmp_path),
    )

    assert code == 3
    # A failed replay must not leave a misleading session behind.
    assert not os.path.exists(session)


def test_main_reads_target_code_from_file(tmp_path, monkeypatch):
    child = _write(str(tmp_path / "child.py"), _BUSY_CHILD)
    markers = str(tmp_path / "markers.json")
    session = str(tmp_path / "session.zip")
    code_file = _write(str(tmp_path / "target.txt"), "answer <- 42")

    argv = [
        "--run",
        json.dumps([sys.executable, child, markers, "0.2"]),
        "--session",
        session,
        "--markers",
        markers,
        "--interval",
        "0.05",
        "--prefix-count",
        "1",
        "--work-dir",
        str(tmp_path),
        "--target-code-file",
        code_file,
        "--language",
        "r",
    ]

    assert harness.main(argv) == 0
    history = _read_cell_history(session)
    assert history.iloc[1]["raw_cell"] == "answer <- 42"
