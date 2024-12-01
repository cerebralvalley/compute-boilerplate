from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
import torch
from config import Config
from huggingface_hub import login

class ModelManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"API Using Device: {self.device}")

        if Config.HUGGINGFACE_ACCESS_TOKEN:
            login(token=Config.HUGGINGFACE_ACCESS_TOKEN)
        self.use_auth = bool(Config.HUGGINGFACE_ACCESS_TOKEN)

        self.is_multimodal = hasattr(AutoModelForCausalLM.from_pretrained(Config.MODEL_NAME, use_auth_token=self.use_auth), "vision_tower")

        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, use_auth_token=self.use_auth)
        self.model = AutoModelForCausalLM.from_pretrained(Config.MODEL_NAME, use_auth_token=self.use_auth).to(self.device)
        self.processor = AutoProcessor.from_pretrained(Config.MODEL_NAME, use_auth_token=self.use_auth) if self.is_multimodal else None

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model

    def get_processor(self):
        return self.processor

    def is_multimodal_model(self):
        return self.is_multimodal

    def get_device(self):
        return self.device
    