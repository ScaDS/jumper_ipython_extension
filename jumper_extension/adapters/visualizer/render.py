"""Renderer registry infrastructure.

Defines the PlotResult/SeriesItem contract and the RENDERERS dict.
Actual renderer implementations live in renderers.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, TypedDict

if TYPE_CHECKING:
    import pandas as pd


class SeriesItem(TypedDict):
    label: str
    data: "pd.Series"
    color: str
    width: float
    opacity: float
    linestyle: str  # "solid" | "dashed" | "dotted"


class PlotResult(TypedDict):
    series: list[SeriesItem]
    title: str
    ylim: Optional[tuple[float, float]]  # None → backend computes from data


RENDERERS: dict[str, Callable] = {}


def register(plot_type: str) -> Callable:
    """Register a renderer function for a given plot type.

    Usage::

        from jumper_extension.adapters.visualizer.render import register

        @register("my_type")
        def render_my_type(df, config, level, hardware, io_window):
            ...
            return PlotResult(series=[...], title=config.title, ylim=None)
    """
    def decorator(fn: Callable) -> Callable:
        RENDERERS[plot_type] = fn
        return fn
    return decorator
