
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

    # Gemini Configuration
    gemini_api_key: str = ""
    gemini_chat_model: str = "gemini-2.5-flash"
    gemini_embed_model: str = "gemini-embedding-001"

    # Embedding Configuration
    embed_dim: int = 768

    # AWS Configuration (temporarily retained for S3)
    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    s3_bucket: str = ""

    # Database
    database_url: str = ""

    # Pinecone (temporarily retained)
    pinecone_api_key: str = ""
    pinecone_index_name: str = "atlas-enterprise-ai-index"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_ready_timeout: int = 180

    # RAG Configuration
    max_upload_bytes: int = 10485760
    min_relevance: float = 0.25

    # Memory and HITL
    hitl_enabled: bool = True
    memory_turns: int = 6

    # Persistent Storage
    chroma_path: str = "./chroma_db"
    upload_dir: str = "./uploads"


@lru_cache
def settings() -> Settings:
    return Settings()
