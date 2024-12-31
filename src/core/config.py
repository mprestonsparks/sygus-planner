"""Configuration management using Pydantic."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings.
    
    Following FastAPI's best practices for configuration management:
    https://fastapi.tiangolo.com/advanced/settings/
    """
    anthropic_api_key: str
    model_name: str = "claude-3-opus-20240229"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
