from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    OPEN_AI_KEY: str
    GOOGLE_AI_KEY: str
    GROQ_AI_KEY: str

    model_config = SettingsConfigDict(env_file=".env")

config = Config()