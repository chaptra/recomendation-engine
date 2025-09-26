"""Configuration management for the recommendation engine."""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600


@dataclass
class CacheConfig:
    """Cache configuration."""
    directory: str = "./cache"
    ttl_hours: int = 24
    max_size_gb: float = 5.0


@dataclass
class ModelConfig:
    """ML model configuration."""
    max_features: int = 5000
    min_df: int = 3
    max_df: float = 0.7
    ngram_range: tuple = (1, 2)
    description_max_length: int = 500
    similarity_threshold: float = 0.1


@dataclass
class APIConfig:
    """API configuration."""
    max_recommendations: int = 50
    default_recommendations: int = 10
    rate_limit_per_minute: int = 100
    timeout_seconds: int = 30


@dataclass
class Config:
    """Main configuration class."""
    database: DatabaseConfig
    cache: CacheConfig
    model: ModelConfig
    api: APIConfig
    log_level: str = "INFO"
    environment: str = "development"


def load_config() -> Config:
    """Load configuration from environment variables."""
    return Config(
        database=DatabaseConfig(
            url=os.getenv("RDS_DB_URL", ""),
            pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
            pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
            pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "3600"))
        ),
        cache=CacheConfig(
            directory=os.getenv("CACHE_DIR", "./cache"),
            ttl_hours=int(os.getenv("CACHE_TTL_HOURS", "24")),
            max_size_gb=float(os.getenv("CACHE_MAX_SIZE_GB", "5.0"))
        ),
        model=ModelConfig(
            max_features=int(os.getenv("MODEL_MAX_FEATURES", "5000")),
            min_df=int(os.getenv("MODEL_MIN_DF", "3")),
            max_df=float(os.getenv("MODEL_MAX_DF", "0.7")),
            description_max_length=int(os.getenv("MODEL_DESC_MAX_LENGTH", "500")),
            similarity_threshold=float(os.getenv("MODEL_SIM_THRESHOLD", "0.1"))
        ),
        api=APIConfig(
            max_recommendations=int(os.getenv("API_MAX_RECOMMENDATIONS", "50")),
            default_recommendations=int(os.getenv("API_DEFAULT_RECOMMENDATIONS", "10")),
            rate_limit_per_minute=int(os.getenv("API_RATE_LIMIT", "100")),
            timeout_seconds=int(os.getenv("API_TIMEOUT_SECONDS", "30"))
        ),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        environment=os.getenv("ENVIRONMENT", "development")
    )