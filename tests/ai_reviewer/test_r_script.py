"""Tests for the R replay script builder (pure Python, no R needed)."""
from jumper_extension.adapters.ai_reviewer.language import r_script
from jumper_extension.adapters.ai_reviewer.language.r_script import build_r_script


def test_names_vector_renders_empty_and_populated():
    assert r_script._names_vector([]) == "character(0)"
    assert r_script._names_vector(["a", "b"]) == 'c("a", "b")'


def test_r_string_escapes_quotes_and_backslashes():
    assert r_script._r_string('a"b\\c') == '"a\\"b\\\\c"'


def test_build_r_script_writes_prefix_target_markers_and_fingerprints(tmp_path):
    path = build_r_script(
        prefix_cells=[{"index": 0, "raw_cell": "base <- 10"}],
        target_code="y <- base + 1",
        output_names=["y"],
        markers_path=str(tmp_path / "m.json"),
        fingerprint_path=str(tmp_path / "f.json"),
        output_path=str(tmp_path / "script.R"),
    )
    assert path.endswith("script.R")
    text = open(path, encoding="utf-8").read()

    # Prefix runs before the timed region, then the target is bracketed.
    assert "base <- 10" in text
    assert text.index("base <- 10") < text.index(".jumper_start")
    assert "y <- base + 1" in text
    assert ".jumper_end <- as.numeric(Sys.time())" in text
    # Both artifact files are written by the script itself.
    assert str(tmp_path / "m.json") in text
    assert str(tmp_path / "f.json") in text
    assert 'c("y")' in text


def test_build_r_script_skips_pure_magic_prefix_cells(tmp_path):
    path = build_r_script(
        prefix_cells=[
            {"index": 0, "raw_cell": "%wrap_kernel ir\n%perfmonitor_fast_setup"},
            {"index": 1, "raw_cell": "base <- 10"},
        ],
        target_code="y <- base + 1",
        output_names=["y"],
        markers_path=str(tmp_path / "m.json"),
        fingerprint_path=str(tmp_path / "f.json"),
        output_path=str(tmp_path / "s.R"),
    )
    text = open(path, encoding="utf-8").read()
    # The magic cell leaves no trace; the real R prefix survives.
    assert "%wrap_kernel" not in text
    assert "%perfmonitor_fast_setup" not in text
    assert "base <- 10" in text
    assert "# --- cell 0 ---" not in text
    assert "# --- cell 1 ---" in text


def test_build_r_script_only_fingerprints_cell_bindings(tmp_path):
    # Names are looked up in the global env with inherits = FALSE, so a name
    # that shadows a base-R object is never fingerprinted unless the cell bound it.
    path = build_r_script(
        prefix_cells=[],
        target_code="x <- 1",
        output_names=["x"],
        markers_path=str(tmp_path / "m.json"),
        fingerprint_path=str(tmp_path / "f.json"),
        output_path=str(tmp_path / "s.R"),
    )
    text = open(path, encoding="utf-8").read()
    assert "envir = globalenv(), inherits = FALSE" in text
