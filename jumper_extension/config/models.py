"""Pydantic v2 models for plot-subset descriptors.

``MetricConfig`` is a discriminated union on the ``type`` field; Pydantic
selects the correct concrete model automatically during ``model_validate()``.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field


class SeriesStyle(BaseModel):
    column: str
    label: str
    color: str = "steelblue"
    width: float = 2.0
    y_axis: Literal["left", "right"] = "left"


class SingleSeriesConfig(BaseModel):
    type: Literal["single_series"]
    column: str
    title: str
    label: str
    ylim: Optional[tuple[float, float]] = None


class SummarySeriesConfig(BaseModel):
    type: Literal["summary_series"]
    columns: list[str]
    title: str
    label: str
    ylim: Optional[tuple[float, float]] = None


class MultiSeriesConfig(BaseModel):
    type: Literal["multi_series"]
    prefix: str
    title: str
    label: str
    ylim: Optional[tuple[float, float]] = None


class CompositeSeriesConfig(BaseModel):
    type: Literal["composite_series"]
    series: list[SeriesStyle]
    title: str
    label: str
    ylim: Optional[tuple[float, float]] = None


MetricConfig = Annotated[
    Union[
        SingleSeriesConfig,
        SummarySeriesConfig,
        MultiSeriesConfig,
        CompositeSeriesConfig,
    ],
    Field(discriminator="type"),
]

def validate_metric_config(data: dict):
    from pydantic import TypeAdapter
    return TypeAdapter(MetricConfig).validate_python(data)


class PerfReportsDefaults(BaseModel):
    level: str = "process"
    text: bool = False


class MonitoringDefaults(BaseModel):
    default_interval: float = 1.0
    live_update_interval: float = 2.0
    live_window_seconds: float = 120.0


class ExportVarsConfig(BaseModel):
    perfdata: str = "perfdata_df"
    cell_history: str = "cell_history_df"


class LoadedVarsConfig(BaseModel):
    perfdata: str = "loaded_perfdata_df"
    cell_history: str = "loaded_cell_history_df"


class SettingsConfig(BaseModel):
    perfreports: PerfReportsDefaults = Field(default_factory=PerfReportsDefaults)
    monitoring: MonitoringDefaults = Field(default_factory=MonitoringDefaults)
    export_vars: ExportVarsConfig = Field(default_factory=ExportVarsConfig)
    loaded_vars: LoadedVarsConfig = Field(default_factory=LoadedVarsConfig)
    visualizer_backend: str = "plotly"


class PlotsConfig(BaseModel):
    default_subsets: list[str] = Field(default_factory=lambda: ["cpu", "mem", "io"])
    subsets: dict[str, dict[str, MetricConfig]]


class AIConfig(BaseModel):
    base_url: str = "https://llm.scads.ai/v1"
    model: str = "MiniMaxAI/MiniMax-M2.7"
    max_tokens: int = 8000
    timeout: float = 120.0
    max_retries: int = 2
    streaming: bool = False
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    enable_thinking: bool | None = None
    extra_body: dict = Field(default_factory=dict)
    api_key_env: str = "JUMPER_AI_API_KEY"


class PythonCollectorsConfig(BaseModel):
    collectors: dict[str, dict]


class CCollectorsConfig(BaseModel):
    collectors: list[str]


class CollectorsConfig(BaseModel):
    python: PythonCollectorsConfig
    c: CCollectorsConfig


class AppConfig(BaseModel):
    settings: SettingsConfig = Field(default_factory=SettingsConfig)
    plots: PlotsConfig
    ai: AIConfig = Field(default_factory=AIConfig)
    collectors: CollectorsConfig
