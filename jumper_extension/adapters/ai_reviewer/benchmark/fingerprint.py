"""Approximate signatures of a cell's results, so a variant that quietly
computes something else is not reported as a speedup.

Signatures are statistical (shape/dtype/mean/std/min/max), never exact hashes:
reordering a sum - which is what vectorising *does* - changes the last bits of a
float, so byte equality would flag every honest optimization as wrong.
"""
import ast
import json
import math

REL_TOL = 1e-6

MATCH = "match"
DIFFERS = "differs"
UNVERIFIED = "unverified"


def assigned_names(code: str) -> list[str]:
    """Top-level names *code* assigns, in source order."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        else:
            continue
        for target in targets:
            names.extend(_target_names(target))
    return list(dict.fromkeys(names))


def _target_names(target) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names = []
        for element in target.elts:
            names.extend(_target_names(element))
        return names
    # x.attr = ... / x[i] = ... bind nothing new
    return []


def fingerprint(value) -> dict | None:
    """Statistical signature of *value*, or None when it cannot be summarised."""
    if value is None or isinstance(value, bool):
        return {"kind": "scalar", "value": value}
    if isinstance(value, (int, float)):
        return {"kind": "scalar", "value": float(value)}
    if isinstance(value, str):
        return {"kind": "text", "len": len(value)}
    if hasattr(value, "columns") and hasattr(value, "shape"):
        return {
            "kind": "frame",
            "shape": list(value.shape),
            "columns": [str(column) for column in value.columns][:50],
        }
    return _array_fingerprint(value)


def _array_fingerprint(value) -> dict | None:
    """Summarise anything array-shaped: numpy, cupy, torch, ..."""
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    base = {
        "kind": "array",
        "shape": [int(dimension) for dimension in shape],
        "dtype": str(getattr(value, "dtype", "")),
    }
    try:
        base.update(
            mean=float(value.mean()),
            std=float(value.std()),
            min=float(value.min()),
            max=float(value.max()),
        )
    except Exception:
        # Non-numeric or exotic array: shape and dtype still catch a variant
        # that silently processed fewer elements.
        pass
    return base


def capture(names: list[str], namespace: dict) -> dict:
    """Fingerprint each of *names* present in *namespace*."""
    captured = {}
    for name in names:
        if name not in namespace:
            continue
        try:
            captured[name] = fingerprint(namespace[name])
        except Exception:
            captured[name] = None
    return captured


def dump(names: list[str], namespace: dict, path: str) -> None:
    """Write fingerprints of *names* to *path*; called from the replay script."""
    with open(path, "w") as handle:
        json.dump(capture(names, namespace), handle)


def load(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def compare(baseline: dict | None, variant: dict | None, rel_tol: float = REL_TOL) -> str:
    """Compare two fingerprints: MATCH, DIFFERS or UNVERIFIED."""
    if not baseline or not variant:
        return UNVERIFIED
    if baseline.get("kind") != variant.get("kind"):
        return DIFFERS

    kind = baseline["kind"]
    if kind == "scalar":
        return _compare_numbers(baseline["value"], variant["value"], rel_tol)
    if kind == "text":
        return MATCH if baseline["len"] == variant["len"] else DIFFERS
    if kind == "frame":
        same = (
            baseline["shape"] == variant["shape"]
            and baseline["columns"] == variant["columns"]
        )
        return MATCH if same else DIFFERS
    return _compare_arrays(baseline, variant, rel_tol)


def _compare_numbers(baseline, variant, rel_tol: float) -> str:
    if isinstance(baseline, bool) or baseline is None or isinstance(variant, bool) or variant is None:
        return MATCH if baseline == variant else DIFFERS
    if math.isnan(baseline) and math.isnan(variant):
        return MATCH
    return MATCH if math.isclose(baseline, variant, rel_tol=rel_tol) else DIFFERS


def _compare_arrays(baseline: dict, variant: dict, rel_tol: float) -> str:
    if baseline["shape"] != variant["shape"]:
        return DIFFERS
    stats = ("mean", "std", "min", "max")
    if not all(stat in baseline and stat in variant for stat in stats):
        # Shape matched but the values were never summarised.
        return UNVERIFIED
    for stat in stats:
        if _compare_numbers(baseline[stat], variant[stat], rel_tol) == DIFFERS:
            return DIFFERS
    return MATCH


def compare_all(baseline: dict, variant: dict, rel_tol: float = REL_TOL) -> tuple[str, list[str]]:
    """Verdict over every captured name, plus the names that diverged."""
    if not baseline:
        return UNVERIFIED, []

    differing = []
    verdict = MATCH
    for name, expected in baseline.items():
        result = compare(expected, variant.get(name), rel_tol)
        if result == DIFFERS:
            differing.append(name)
        elif result == UNVERIFIED and verdict == MATCH:
            verdict = UNVERIFIED
    if differing:
        return DIFFERS, differing
    return verdict, []
