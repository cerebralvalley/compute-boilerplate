import dotenv
import os

dotenv.load_dotenv()

class Config:
    MODEL_NAME: str = "microsoft/phi-1_5"
    PORT: int = 8000
    DEFAULT_MAX_TOKENS: int = 100
    DEFAULT_TEMPERATURE: float = 1
    HUGGINGFACE_ACCESS_TOKEN: str = os.getenv("HUGGINGFACE_ACCESS_TOKEN", "")
