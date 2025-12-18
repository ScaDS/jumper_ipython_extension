from typing import FrozenSet
from functools import lru_cache

from IPython import get_ipython


@lru_cache(maxsize=1)
def get_line_magics_cached() -> FrozenSet[str]:
    ip = get_ipython()
    return frozenset(ip.magics_manager.lsmagic().get("line", []))


def is_known_line_magic(line: str, line_magics: frozenset) -> bool:
    # Allow leading spaces
    s = line.lstrip()
    if not s.startswith("%"):
        return False
    # Extract the magic name between '%' and the first space or end of line
    name = s[1:].split(None, 1)[0]
    return name in line_magics


def is_pure_line_magic_cell(raw_cell: str) -> bool:
    """
    A pure line-magic cell = each non-empty line is either:
      - starts with %<known_magic> (optionally with arguments),
      - or is a comment (#...).
    """
    line_magics = get_line_magics_cached()
    # Get the list of available line magics, names without '%'
    lines = raw_cell.splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            # skip empty lines
            continue
        if stripped.startswith("#"):
            # skip comments
            continue
        if is_known_line_magic(line, line_magics):
            # skip line magic
            continue
        # any other non-empty line is considered code -> not "pure"
        return False
    return True


def get_called_line_magics(raw_cell: str) -> list:
    """
    Get the list of line magics called in a cell
    """
    line_magics = get_line_magics_cached()
    called_line_magics = []
    # Get the list of available line magics, names without '%'
    lines = raw_cell.splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            # skip empty lines
            continue
        if stripped.startswith("#"):
            # skip comments
            continue
        if is_known_line_magic(line, line_magics):
            called_line_magics.append(stripped[1:])
    return called_line_magics