from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    tavily_api_key: str = ""

    database_url: str = "postgresql+asyncpg://kresearch:kresearch@localhost:5433/kresearch"

    default_task_budget_usd: float = 1.0


settings = Settings()
