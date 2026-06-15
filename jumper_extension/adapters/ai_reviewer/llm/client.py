import os
from dataclasses import dataclass

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
        )


def build_llm(config: LLMClientConfig) -> BaseChatModel:
    """Build a chat model client from *config*.

    Uses ``langchain-openai``'s ``ChatOpenAI`` with a custom ``base_url``
    so any OpenAI-compatible endpoint (e.g. SCADS) can be targeted.
    """
    return ChatOpenAI(
        base_url=config.base_url,
        api_key=config.api_key,
        model=config.model,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
    )
