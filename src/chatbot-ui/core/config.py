from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    OPEN_AI_KEY: str
    GOOGLE_AI_KEY: str
    GROQ_AI_KEY: str
    QDRANT_URL: str
    QDRANT_COLLECTION_NAME: str
    EMBEDDING_MODEL: str
    EMBEDDING_MODEL_PROVIDER: str

    model_config = SettingsConfigDict(env_file=".env")

config = Config()