from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Document & Multimedia Q&A"
    database_url: str = "sqlite+aiosqlite:///./app.db"
    redis_url: str = "redis://redis:6379/0"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "models/gemini-embedding-001"
    vector_db_type: str = "local"
    pinecone_api_key: str | None = None
    transcription_api_key: str | None = None
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 60 * 24
    upload_dir: str = "uploads"
    max_upload_mb: int = 100
    rate_limit_per_minute: int = 60
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    model_config = SettingsConfigDict(env_file=(".env", "../.env"), env_file_encoding="utf-8")

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
