"""Render the R replay the harness runs: prefix + target, timed and fingerprinted.

The counterpart of ``benchmark/script.py`` for R. It writes a plain ``.R`` file
that the shared harness launches with ``Rscript``; unlike the Python replay it
does not touch perfmonitor or the session export - that stays in the harness.
The R side owns only the two seams that must live inside the language runtime:
it brackets the target cell with epoch timestamps (which the harness maps onto
the sampler clock) and dumps a statistical fingerprint of the cell's outputs in
the very schema ``fingerprint.py`` reads, so an R rewrite that quietly computes
something else is caught the same way a Python one is.
"""
from pathlib import Path

# Base-R fingerprint helper, kept dependency-free (no jsonlite): it emits the
# same {"kind": ...} objects fingerprint.py produces for Python values. Baseline
# and variants are always the same language, so R's sample sd only has to agree
# with itself, never with numpy's population std.
_HELPER = '''\
.jumper_num <- function(x) {
  if (length(x) != 1 || is.na(x) || !is.finite(x)) return("null")
  sprintf("%.10g", x)
}
.jumper_fp <- function(value) {
  if (is.data.frame(value)) {
    cols <- colnames(value)
    if (length(cols) > 50) cols <- cols[1:50]
    cols_json <- paste0('"', gsub('"', '\\\\\\\\"', cols), '"', collapse = ", ")
    return(sprintf('{"kind": "frame", "shape": [%d, %d], "columns": [%s]}',
                   nrow(value), ncol(value), cols_json))
  }
  if (is.character(value) && length(value) == 1) {
    return(sprintf('{"kind": "text", "len": %d}', nchar(value)))
  }
  if (is.logical(value) && length(value) == 1) {
    return(sprintf('{"kind": "scalar", "value": %s}',
                   if (isTRUE(value)) "true" else "false"))
  }
  if (is.numeric(value) && length(value) == 1) {
    return(sprintf('{"kind": "scalar", "value": %s}', .jumper_num(as.numeric(value))))
  }
  if (is.numeric(value)) {
    d <- dim(value)
    if (is.null(d)) d <- length(value)
    v <- as.numeric(value)
    return(sprintf('{"kind": "array", "shape": [%s], "dtype": "%s", "mean": %s, "std": %s, "min": %s, "max": %s}',
                   paste(d, collapse = ", "),
                   if (is.integer(value)) "integer" else "double",
                   .jumper_num(mean(v, na.rm = TRUE)),
                   .jumper_num(sd(v, na.rm = TRUE)),
                   .jumper_num(min(v, na.rm = TRUE)),
                   .jumper_num(max(v, na.rm = TRUE))))
  }
  return("null")
}
'''


def _r_string(text: str) -> str:
    """Quote *text* as an R double-quoted string literal."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _names_vector(names: list[str]) -> str:
    """Render *names* as an R character vector, ``character(0)`` when empty."""
    if not names:
        return "character(0)"
    return "c(" + ", ".join(_r_string(name) for name in names) + ")"


def _strip_magic_lines(code: str) -> str:
    """Drop IPython/wrapper magic lines (``%...``) from a replayed cell.

    Under the wrapper kernel a pure-magic cell (``%wrap_kernel``,
    ``%perfmonitor_fast_setup``, ...) is run locally and never forwarded to R,
    so it carries no R state and replaying it as R would be a syntax error. R
    never begins a statement with ``%`` (it only appears in infix operators like
    ``%*%``, mid-expression), so a leading ``%`` unambiguously marks a magic.
    """
    kept = [line for line in code.splitlines() if not line.lstrip().startswith("%")]
    return "\n".join(kept)


def build_r_script(
    prefix_cells: list[dict],
    target_code: str,
    output_names: list[str],
    markers_path: str,
    fingerprint_path: str,
    output_path: str,
) -> str:
    """Write the R replay script and return its path.

    The prefix cells run first to rebuild the state the target needs, exactly as
    the Python replay does; the target is bracketed by ``Sys.time()`` so only its
    own cost is timed, and the fingerprint dump happens afterwards so its work
    never lands in the measured window.
    """
    parts = [_HELPER]
    for cell in prefix_cells:
        body = _strip_magic_lines(cell["raw_cell"])
        if not body.strip():
            continue  # a pure-magic cell contributes no R state
        parts.append(f"\n# --- cell {cell['index']} ---\n")
        parts.append(body)
        parts.append("\n")
    parts.append("\n# --- cell under test ---\n")
    parts.append(".jumper_start <- as.numeric(Sys.time())\n")
    parts.append(target_code)
    parts.append("\n.jumper_end <- as.numeric(Sys.time())\n")
    parts.append(
        "\n# --- timing markers ---\n"
        f".jumper_mh <- file({_r_string(markers_path)})\n"
        'writeLines(sprintf(\'{"start": %.6f, "end": %.6f}\', '
        ".jumper_start, .jumper_end), .jumper_mh)\n"
        "close(.jumper_mh)\n"
    )
    parts.append(
        "\n# --- output fingerprints ---\n"
        f".jumper_names <- {_names_vector(output_names)}\n"
        ".jumper_parts <- character(0)\n"
        "for (.jumper_nm in .jumper_names) {\n"
        # inherits = FALSE on the global env: fingerprint only what the cell
        # actually bound, never a base-R object of the same name (df, t, ...).
        "  if (exists(.jumper_nm, envir = globalenv(), inherits = FALSE)) {\n"
        "    .jumper_val <- get(.jumper_nm, envir = globalenv(), inherits = FALSE)\n"
        "    .jumper_parts <- c(.jumper_parts, paste0('\"', .jumper_nm, "
        "'\": ', .jumper_fp(.jumper_val)))\n"
        "  }\n"
        "}\n"
        f".jumper_fh <- file({_r_string(fingerprint_path)})\n"
        'writeLines(paste0("{", paste(.jumper_parts, collapse = ", "), "}"), '
        ".jumper_fh)\n"
        "close(.jumper_fh)\n"
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts), encoding="utf-8")
    return str(path)
