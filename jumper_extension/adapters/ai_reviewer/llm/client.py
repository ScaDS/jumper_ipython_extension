import os
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

DEFAULT_BASE_URL = "https://llm.scads.ai/v1"
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


@dataclass
class LLMClientConfig:
    """Configuration for the LLM client used by the AI review agent."""
    base_url: str
    api_key: str
    model: str

    @classmethod
    def from_env(cls) -> "LLMClientConfig":
        """Build a config from ``JUMPER_AI_*`` environment variables."""
        return cls(
            base_url=os.environ.get("JUMPER_AI_BASE_URL", DEFAULT_BASE_URL),
            api_key=os.environ.get("JUMPER_AI_API_KEY", ""),
            model=os.environ.get("JUMPER_AI_MODEL", DEFAULT_MODEL),
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
    )
