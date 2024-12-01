from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
from config import Config
from utils import check_system_resources, serialize
from llama_stack.apis.inference.inference import (
    ChatCompletionResponse,
    ChatCompletionResponseStreamChunk,
    ChatCompletionRequest,
    StopReason,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseStreamChunk
)
from typing import Union
from model import ModelManager
from processor import InputProcessor

app = FastAPI()

check_system_resources(Config.MODEL_NAME)

model_manager = ModelManager()
input_processor = InputProcessor(model_manager)

@app.post("/inference/completion")
async def completion(
    request: CompletionRequest
) -> Union[CompletionResponse, CompletionResponseStreamChunk]:
    """
    Inferences completion for your chosen huggingface model, with Image inputs allowed!
    """
    max_tokens = min(request.sampling_params.max_tokens or float('inf'), Config.DEFAULT_MAX_TOKENS)
    temperature = request.sampling_params.temperature or Config.DEFAULT_TEMPERATURE

    if request.stream:
        async def stream_generator():
            async for token in input_processor.generate_tokens(request.content, max_tokens, temperature):
                yield serialize(CompletionResponseStreamChunk(delta=token)).encode('utf-8') + b'\n'
        return StreamingResponse(stream_generator(), media_type="application/json")
    else:
        output_text = ""
        async for token in input_processor.generate_tokens(request.content, max_tokens, temperature):
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
    Inferences chat completion for your chosen huggingface model, with Image inputs allowed!
    """
    input_content = request.messages[-1].content
    max_tokens = min(request.sampling_params.max_tokens or float('inf'), Config.DEFAULT_MAX_TOKENS)
    temperature = request.sampling_params.temperature or Config.DEFAULT_TEMPERATURE

    if request.stream:
        async def stream_generator():
            async for token in input_processor.generate_tokens(input_content, max_tokens, temperature):
                yield serialize(ChatCompletionResponseStreamChunk(event={"event_type": "progress", "delta": token})).encode('utf-8') + b'\n'
        return StreamingResponse(stream_generator(), media_type="application/json")
    else:
        output_text = ""
        async for token in input_processor.generate_tokens(input_content, max_tokens, temperature):
            output_text += token
        return ChatCompletionResponse(completion_message={
            "role": "assistant",
            "content": output_text,
            "stop_reason": StopReason.out_of_tokens if len(output_text) >= max_tokens else StopReason.end_of_message
        })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=Config.PORT)
