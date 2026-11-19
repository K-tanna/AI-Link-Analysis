from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration comes from environment variables / .env (never hard-code keys)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM ---
    llm_provider: str = "gemini"  # openai | anthropic | gemini
    llm_model: str = "gemini-flash-lite-latest"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    gemini_thinking_budget: Optional[int] = 0  # 0 to disable thinking (saves tokens/latency), 1024 balanced, -1 unconstrained

    # Optional fallback used only if the primary provider fails after retries
    fallback_provider: str = "gemini"
    fallback_model: str = "gemini-3.1-flash-lite"

    llm_timeout_seconds: float = 90.0
    llm_max_retries: int = 3
    llm_max_output_tokens: int = 8192

    # Optional per-1M-token prices (USD) so the API can report estimated cost per request
    llm_input_price_per_m: Optional[float] = None
    llm_output_price_per_m: Optional[float] = None

    # --- Website fetching ---
    fetch_timeout_seconds: float = 10.0
    fetch_max_bytes: int = 2_000_000
    fetch_max_redirects: int = 5
    max_extra_pages: int = 2
    max_content_chars: int = 12_000  # ~3k tokens of page text sent to the LLM
    user_agent: str = "Mozilla/5.0 (compatible; LinkAnalysisBot/1.0; +https://example.com/bot)"

    # --- Cache ---
    cache_ttl_seconds: int = 3600
    cache_max_entries: int = 256


@lru_cache
def get_settings() -> Settings:
    return Settings()
