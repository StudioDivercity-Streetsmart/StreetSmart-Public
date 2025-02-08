import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings 
from functools import lru_cache

load_dotenv()  # Load environment variables from .env file

class ChatGptSettings(BaseSettings):
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_MAX_TOKENS: int = 2000
    OPENAI_TEMPERATURE: float = 0.7
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_ai_settings():
    return ChatGptSettings()
