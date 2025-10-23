"""Module containing parser utilities for the JUmPER extension."""
import argparse
import shlex
from dataclasses import dataclass
from typing import Optional, Tuple, List, Any

from jumper_extension.utilities import get_available_levels


@dataclass
class ArgParsers:
    """Configuration for command-line argument parsers."""
    perfreport: argparse.ArgumentParser
    auto_perfreports: argparse.ArgumentParser
    export_perfdata: argparse.ArgumentParser
    export_cell_history: argparse.ArgumentParser


def build_perfreport_parser() -> argparse.ArgumentParser:
    """Build an ArgumentParser instance for JUmPER commands."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--cell",
        type=str,
        help="Cell index or range (e.g., 5, 2:8, :5)"
    )
    parser.add_argument(
        "--level",
        default="process",
        choices=get_available_levels(),
        help="Performance level",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Show report in text format"
    )
    return parser

def build_auto_perfreports_parser() -> argparse.ArgumentParser:
    parser = build_perfreport_parser()
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Interval between automatic reports (default: 1 second)",
    )
    return parser

def build_export_perfdata_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--file", type=str, help="Output filename")
    parser.add_argument(
        "--level",
        default="process",
        choices=get_available_levels(),
        help="Performance level",
    )
    return parser

def build_export_cell_history_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--file", type=str, help="Output filename")
    return parser

def parse_arguments(parser: argparse.ArgumentParser, line: str) -> Optional[argparse.Namespace]:
    """Parse common command line arguments for JUmPER commands.
    
    Args:
        line: The command line string to parse
        parser: Optional existing ArgumentParser instance
        
    Returns:
        Parsed arguments or None if parsing failed
    """
    try:
        args = (
            parser.parse_args(shlex.split(line))
            if line
            else parser.parse_args([])
        )
    except Exception:
        args = None
    return args


def parse_cell_range(cell_str: str, cell_history: List[Any]) -> Optional[Tuple[int, int]]:
    """Parse a cell range string into start and end indices.
    
    Args:
        cell_str: String representing cell range (e.g., "1:3", "5", ":10")
        cell_history: List of cell history entries to validate indices against
        
    Returns:
        Tuple of (start_idx, end_idx) or None if invalid
    """
    if not cell_str:
        return None
        
    try:
        max_idx = len(cell_history) - 1
        if ":" in cell_str:
            start_str, end_str = cell_str.split(":", 1)
            start_idx = 0 if not start_str else int(start_str)
            end_idx = max_idx if not end_str else int(end_str)
        else:
            start_idx = end_idx = int(cell_str)
            
        if 0 <= start_idx <= end_idx <= max_idx:
            return start_idx, end_idx
    except (ValueError, IndexError, AttributeError):
        pass
        
    return None
