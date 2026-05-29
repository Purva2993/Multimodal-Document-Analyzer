"""
  config.py

  Central configuration for the Multimodal Document Analyzer.
  All environment variables are loaded and validated here.
  Every other module imports settings from this file — nothing
  reads .env directly.
  """

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Pydantic validates types and raises a clear error on startup
    if a required variable is missing or has the wrong type.
    """
    # HuggingFace
    huggingface_token: str = Field(..., env="HUGGINGFACE_TOKEN")
    # Model
    base_model: str = Field("microsoft/layoutlmv3-base", env="BASE_MODEL")
    fine_tuned_model_path: str = Field("./models/fine_tuned", env="FINE_TUNED_MODEL_PATH")
    confidence_threshold: float = Field(0.5, env="CONFIDENCE_THRESHOLD")
    # ChromaDB
    chroma_persist_dir: str = Field("./data/chroma_db", env="CHROMA_PERSIST_DIR")
    # Dataset
    dataset_name: str = Field("nielsr/docvqa_1200_examples", env="DATASET_NAME")
    raw_data_dir: str = Field("./data/raw", env="RAW_DATA_DIR")
    processed_data_dir: str = Field("./data/processed", env="PROCESSED_DATA_DIR")
    # Training
    num_epochs: int = Field(5, env="NUM_EPOCHS")
    batch_size: int = Field(4, env="BATCH_SIZE")
    learning_rate: float = Field(5e-5, env="LEARNING_RATE")
    max_seq_length: int = Field(512, env="MAX_SEQ_LENGTH")
    # Enterprise (optional)
    azure_doc_intelligence_endpoint: str = Field("", env="AZURE_DOC_INTELLIGENCE_ENDPOINT")
    azure_doc_intelligence_key: str = Field("", env="AZURE_DOC_INTELLIGENCE_KEY")
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

def get_settings() -> Settings:
    """
    Returns the application settings instance.
    Called at the top of each module that needs config values.
    Returns:
        Settings: validated settings object loaded from .env
    """
    return Settings()