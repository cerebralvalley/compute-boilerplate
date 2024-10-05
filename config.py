import dotenv
import os

dotenv.load_dotenv()

class Config:
    MODEL_NAME: str = "mistralai/Mistral-7B-Instruct-v0.2"
    PORT: int = 8000
    MAX_TOKENS: int = 100
    HUGGINGFACE_ACCESS_TOKEN: str = os.getenv("HUGGINGFACE_ACCESS_TOKEN", "")
