import torch
from typing import Union, AsyncGenerator, List
from fastapi import HTTPException
from PIL import Image
import requests
from io import BytesIO
from llama_models.llama3.api.datatypes import ImageMedia
from model import ModelManager

class InputProcessor:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def process_image(self, image_media: ImageMedia) -> torch.Tensor:
        """
        Processes the image into a Tensor so we can use it if the model is multimodal.
        """
        if not self.model_manager.is_multimodal_model():
            raise HTTPException(status_code=400, detail="This model does not support image inputs.")
        
        if isinstance(image_media.image, str):  # URL case
            response = requests.get(image_media.image)
            image = Image.open(BytesIO(response.content))
        else:  # PIL Image case
            image = image_media.image
        
        processor = self.model_manager.get_processor()
        processed_image = processor(images=image, return_tensors="pt")["pixel_values"].to(self.model_manager.get_device())
        return processed_image

    def prepare_input(self, content: Union[str, ImageMedia, List[Union[str, ImageMedia]]]) -> dict:
        """
        Prepares the input into the model for ideal inference. 
        """
        model_inputs = {}
        tokenizer = self.model_manager.get_tokenizer()
        device = self.model_manager.get_device()
        
        if isinstance(content, str):
            model_inputs["input_ids"] = tokenizer.encode(content, return_tensors="pt").to(device)
            model_inputs["attention_mask"] = torch.ones_like(model_inputs["input_ids"]).to(device)
            return model_inputs
        
        if self.model_manager.is_multimodal_model():
            if isinstance(content, ImageMedia):
                model_inputs["pixel_values"] = self.process_image(content)
                return model_inputs
            
            text_parts = []
            image_features_list = []
            
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, ImageMedia):
                    image_features = self.process_image(item)
                    if image_features is not None:
                        image_features_list.append(image_features)
            
            if text_parts:
                model_inputs["input_ids"] = tokenizer.encode(" ".join(text_parts), return_tensors="pt").to(device)
                model_inputs["attention_mask"] = torch.ones_like(model_inputs["input_ids"]).to(device)
            
            if image_features_list:
                model_inputs["pixel_values"] = torch.cat(image_features_list, dim=0)
        else:  # for text-only models, extract only text content
            if isinstance(content, ImageMedia):
                raise HTTPException(status_code=400, detail="This model does not support image inputs.")
            elif isinstance(content, str):
                text = content
            else:
                text = " ".join([item for item in content if isinstance(item, str)])
                if any(isinstance(item, ImageMedia) for item in content):
                    raise HTTPException(status_code=400, detail="This model does not support image inputs.")
            
            model_inputs["input_ids"] = tokenizer.encode(text, return_tensors="pt").to(device)
            model_inputs["attention_mask"] = torch.ones_like(model_inputs["input_ids"]).to(device)
        
        return model_inputs

    async def generate_tokens(
        self,
        content: Union[str, ImageMedia, List[Union[str, ImageMedia]]], 
        max_tokens: int, 
        temperature: float
    ) -> AsyncGenerator[str, None]:
        """
        Generates the tokens and does the actual inference!
        """
        model_inputs = self.prepare_input(content)
        model = self.model_manager.get_model()
        tokenizer = self.model_manager.get_tokenizer()
        device = self.model_manager.get_device()
        past = None
        
        for _ in range(max_tokens):
            with torch.no_grad():
                if past is not None:
                    model_inputs["past_key_values"] = past
                model_inputs["use_cache"] = True
                
                outputs = model(**model_inputs)
                logits = outputs.logits[:, -1, :]
        
                scaled_logits = logits / max(temperature, 1e-8)
                probs = torch.softmax(scaled_logits, dim=-1)
                token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                
                token_str = tokenizer.decode(token.item())
                yield token_str

                if token.item() == tokenizer.eos_token_id:
                    break
                
                model_inputs = {
                    "input_ids": token.unsqueeze(0),
                    "attention_mask": torch.ones(1, 1).to(device)
                }
                past = outputs.past_key_values
