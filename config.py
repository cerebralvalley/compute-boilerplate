import dotenv
import os

dotenv.load_dotenv()

class Config:
    MODEL_NAME: str = "openai-community/gpt2"
    PORT: int = 8000
    MAX_TOKENS: int = 100
    HUGGINGFACE_ACCESS_TOKEN: str = os.getenv("HUGGINGFACE_ACCESS_TOKEN", "")
    