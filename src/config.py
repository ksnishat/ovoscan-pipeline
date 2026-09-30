"""
OvoScan Configuration
Pydantic settings for environment-based configuration
"""

from functools import lru_cache
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "ovoscan"
    app_env: str = Field(default="development", alias="APP_ENV")
    app_log_level: str = Field(default="INFO", alias="APP_LOG_LEVEL")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_workers: int = Field(default=2, alias="API_WORKERS")

    # Model
    model_path: str = Field(default="models/best.pt", alias="MODEL_PATH")
    confidence_threshold: float = Field(default=0.25, alias="CONFIDENCE_THRESHOLD")
    iou_threshold: float = Field(default=0.45, alias="IOU_THRESHOLD")
    device: str = Field(default="auto", alias="DEVICE")

    # MLflow
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", alias="MLFLOW_TRACKING_URI")
    mlflow_model_name: str = Field(default="ovoscan-defect-model", alias="MLFLOW_MODEL_NAME")
    mlflow_model_stage: str = Field(default="Production", alias="MLFLOW_MODEL_STAGE")

    # Ollama (Local LLM)
    ollama_host: str = Field(default="http://localhost:11434", alias="OLLAMA_HOST")
    ollama_model: str = Field(default="llama3.2", alias="OLLAMA_MODEL")
    ollama_timeout: int = Field(default=60, alias="OLLAMA_TIMEOUT")

    # RAG Settings
    enable_rag: bool = Field(default=True, alias="ENABLE_RAG")
    rag_top_k: int = Field(default=5, alias="RAG_TOP_K")
    knowledge_base_path: str = Field(default="data/knowledge_base", alias="KNOWLEDGE_BASE_PATH")

    # ZenML
    zenml_tracking_uri: str = Field(default="http://localhost:8080", alias="ZENML_TRACKING_URI")

    # DVC
    dvc_remote: str = Field(default="s3://ovoscan-dvc", alias="DVC_REMOTE")

    # Dash Dashboard
    dash_port: int = Field(default=8050, alias="DASH_PORT")
    dash_debug: bool = Field(default=False, alias="DASH_DEBUG")

    # Monitoring
    enable_metrics: bool = Field(default=True, alias="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, alias="METRICS_PORT")

    # Security
    secret_key: str = Field(default="change-me-in-production", alias="SECRET_KEY")
    cors_origins: List[str] = Field(default=["*"], alias="CORS_ORIGINS")

    # Health Check
    health_check_interval: int = Field(default=30, alias="HEALTH_CHECK_INTERVAL")

    # Sentry
    sentry_dsn: Optional[str] = Field(default=None, alias="SENTRY_DSN")
    sentry_environment: str = Field(default="development", alias="SENTRY_ENVIRONMENT")
    sentry_release: str = Field(default="1.0.0", alias="SENTRY_RELEASE")
    sentry_traces_sample_rate: float = Field(default=0.1, alias="SENTRY_TRACES_SAMPLE_RATE")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Export for easy access
settings = get_settings()