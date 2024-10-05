from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn
from config import Config
from utils import check_system_resources, serialize
from huggingface_hub import login
from llama_stack.apis.inference.inference import (
    ChatCompletionResponse,
    CompletionResponse,
    ChatCompletionResponseStreamChunk,
    CompletionResponseStreamChunk,
    ChatCompletionRequest,
    CompletionRequest,
    StopReason,
)
from typing import Union, AsyncGenerator
from llama_models.llama3.api.datatypes import interleaved_text_media_as_str

app = FastAPI()

check_system_resources(Config.MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"API Using Device: {device}")

if Config.HUGGINGFACE_ACCESS_TOKEN:
    login(token=Config.HUGGINGFACE_ACCESS_TOKEN)
use_auth = bool(Config.HUGGINGFACE_ACCESS_TOKEN)

tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, use_auth_token=use_auth)
model = AutoModelForCausalLM.from_pretrained(Config.MODEL_NAME, use_auth_token=use_auth).to(device)

async def generate_tokens(input_text: str, max_tokens: int, temperature: float) -> AsyncGenerator[str, None]:
    """
    Returns an async generator that streams tokens from your Huggingface LLM as they come.
    The actual logic for inferencing the model is defined here.
    """
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    past = None
    
    for _ in range(max_tokens):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past, use_cache=True)
            logits = outputs.logits[:, -1, :]
    
            scaled_logits = logits / max(temperature, 1e-8)
            probs = torch.softmax(scaled_logits, dim=-1)
            token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            token_str = tokenizer.decode(token.item())
            yield token_str

            if token.item() == tokenizer.eos_token_id:
                break
            
            input_ids = token.unsqueeze(0)
            attention_mask = torch.cat([attention_mask, torch.ones_like(input_ids)], dim=-1)
            past = outputs.past_key_values

@app.post("/inference/completion")
async def completion(
    request: CompletionRequest
) -> Union[CompletionResponse, CompletionResponseStreamChunk]:
    """
    Inferences standard completion with your defined huggingface model.
    """
    input_text = interleaved_text_media_as_str(request.content)
    max_tokens = min(request.sampling_params.max_tokens or float('inf'), Config.DEFAULT_MAX_TOKENS)
    temperature = request.sampling_params.temperature or Config.DEFAULT_TEMPERATURE

    if request.stream:
        async def stream_generator():
            async for token in generate_tokens(input_text, max_tokens, temperature):
                yield serialize(CompletionResponseStreamChunk(delta=token)).encode('utf-8') + b'\n'
        return StreamingResponse(stream_generator(), media_type="application/json")
    else:
        output_text = ""
        async for token in generate_tokens(input_text, max_tokens, temperature):
            output_text += token
        return CompletionResponse(completion_message={
            "content": output_text,
            "stop_reason": StopReason.out_of_tokens if len(output_text) >= max_tokens else StopReason.end_of_message
        })

@app.post("/inference/chat_completion")
async def chat_completion(
    request: ChatCompletionRequest
) -> Union[ChatCompletionResponse, ChatCompletionResponseStreamChunk]:
    """
    Inferences chat completion using your defined Huggingface model.
    """
    input_text = "\n".join([f"{m.role}: {interleaved_text_media_as_str(m.content)}" for m in request.messages])
    max_tokens = min(request.sampling_params.max_tokens or float('inf'), Config.DEFAULT_MAX_TOKENS)
    temperature = request.sampling_params.temperature or Config.DEFAULT_TEMPERATURE

    if request.stream:
        async def stream_generator():
            async for token in generate_tokens(input_text, max_tokens, temperature):
                yield serialize(ChatCompletionResponseStreamChunk(event={"event_type": "progress", "delta": token})).encode('utf-8') + b'\n'
        return StreamingResponse(stream_generator(), media_type="application/json")
    else:
        output_text = ""
        async for token in generate_tokens(input_text, max_tokens, temperature):
            output_text += token
        return ChatCompletionResponse(completion_message={
            "role": "assistant",
            "content": output_text,
            "stop_reason": StopReason.out_of_tokens if len(output_text) >= max_tokens else StopReason.end_of_message
        })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=Config.PORT)
