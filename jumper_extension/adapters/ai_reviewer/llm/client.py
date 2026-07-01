import copy
import os
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from jumper_extension.config.loader import load_config


@dataclass
class LLMClientConfig:
    """Configuration for the LLM client used by the AI review agent."""
    base_url: str
    api_key: str
    model: str
    max_tokens: int
    timeout: float
    max_retries: int = 2
    streaming: bool = False
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    enable_thinking: bool | None = None
    extra_body: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls) -> "LLMClientConfig":
        """Build a config from AppConfig.ai; the API key comes from the
        environment variable named by ``ai.api_key_env``."""
        ai = load_config().ai
        return cls(
            base_url=ai.base_url,
            api_key=os.environ.get(ai.api_key_env, ""),
            model=ai.model,
            max_tokens=ai.max_tokens,
            timeout=ai.timeout,
            max_retries=ai.max_retries,
            streaming=ai.streaming,
            temperature=ai.temperature,
            top_p=ai.top_p,
            seed=ai.seed,
            enable_thinking=ai.enable_thinking,
            extra_body=ai.extra_body,
        )


def _build_extra_body(config: LLMClientConfig) -> dict:
    """Merge ``enable_thinking`` into the vLLM ``extra_body`` passthrough."""
    extra_body = copy.deepcopy(config.extra_body)
    if config.enable_thinking is not None:
        chat_template_kwargs = extra_body.setdefault("chat_template_kwargs", {})
        chat_template_kwargs["enable_thinking"] = config.enable_thinking
    return extra_body


def build_llm(config: LLMClientConfig) -> BaseChatModel:
    """Build a chat model client from *config*.

    Uses ``langchain-openai``'s ``ChatOpenAI`` with a custom ``base_url``
    so any OpenAI-compatible endpoint (e.g. SCADS) can be targeted. Optional
    sampling parameters are only forwarded when set, so unset values fall back
    to the server defaults; vLLM-specific parameters go through ``extra_body``.
    """
    kwargs: dict[str, Any] = {
        "base_url": config.base_url,
        "api_key": config.api_key,
        "model": config.model,
        "max_tokens": config.max_tokens,
        "timeout": config.timeout,
        "max_retries": config.max_retries,
        "streaming": config.streaming,
    }
    if config.temperature is not None:
        kwargs["temperature"] = config.temperature
    if config.top_p is not None:
        kwargs["top_p"] = config.top_p
    if config.seed is not None:
        kwargs["seed"] = config.seed

    extra_body = _build_extra_body(config)
    if extra_body:
        kwargs["extra_body"] = extra_body

    return ChatOpenAI(**kwargs)
