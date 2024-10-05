import requests
from config import Config
import time
from llama_stack.apis.inference.inference import (
    ChatCompletionRequest,
    CompletionRequest,
    SamplingParams,
)
from utils import serialize
import json

default_url = f"http://localhost:{Config.PORT}"

def completion_stream(content, max_tokens=None):
    url = f"{default_url}/inference/completion"
    
    request = CompletionRequest(
        model=Config.MODEL_NAME,
        content=content,
        sampling_params=SamplingParams(max_tokens=max_tokens or Config.DEFAULT_MAX_TOKENS).model_dump(),
        stream=True
    )
    
    response = requests.post(url, data=serialize(request), headers={'Content-Type': 'application/json'}, stream=True)
    
    if response.status_code == 200:
        print("RESPONSE STREAM:")
        total_response = ""
        start_time = time.time()
        token_count = 0
        for line in response.iter_lines(decode_unicode=True):
            if line:
                chunk_data = json.loads(line)
                delta = chunk_data.get('delta', '')
                print(f"{token_count + 1}: {delta}")
                total_response += delta
                token_count += 1
        end_time = time.time()
        print("\nTOTAL RESPONSE:")
        print(total_response)
        
        duration = end_time - start_time
        tokens_per_second = token_count / duration if duration > 0 else 0
        print(f"\nTokens Per Second (TPS): {tokens_per_second:.2f}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

def completion(content, max_tokens=None):
    url = f"{default_url}/inference/completion"
    
    request = CompletionRequest(
        model=Config.MODEL_NAME,
        content=content,
        sampling_params=SamplingParams(max_tokens=max_tokens or Config.DEFAULT_MAX_TOKENS).model_dump(),
        stream=False
    )
    
    response = requests.post(url, data=serialize(request), headers={'Content-Type': 'application/json'})
    
    if response.status_code == 200:
        print("RESPONSE:")
        response_data = response.json()
        generated_text = response_data.get("completion_message", {}).get("content", "")
        print(generated_text)
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

def chat_completion_stream(messages, max_tokens=None):
    url = f"{default_url}/inference/chat_completion"
    
    request = ChatCompletionRequest(
        model=Config.MODEL_NAME,
        messages=messages,
        sampling_params=SamplingParams(max_tokens=max_tokens or Config.DEFAULT_MAX_TOKENS).model_dump(),
        stream=True
    )
    
    response = requests.post(url, data=serialize(request), headers={'Content-Type': 'application/json'}, stream=True)
    
    if response.status_code == 200:
        print("RESPONSE STREAM:")
        total_response = ""
        start_time = time.time()
        token_count = 0
        for line in response.iter_lines(decode_unicode=True):
            if line:
                chunk_data = json.loads(line)
                delta = chunk_data.get('event', {}).get('delta', '')
                print(f"{token_count + 1}: {delta}")
                total_response += delta
                token_count += 1
        end_time = time.time()
        print("\nTOTAL RESPONSE:")
        print(total_response)
        
        duration = end_time - start_time
        tokens_per_second = token_count / duration if duration > 0 else 0
        print(f"\nTokens Per Second (TPS): {tokens_per_second:.2f}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

def chat_completion(messages, max_tokens=None):
    url = f"{default_url}/inference/chat_completion"
    
    request = ChatCompletionRequest(
        model=Config.MODEL_NAME,
        messages=messages,
        sampling_params=SamplingParams(max_tokens=max_tokens or Config.DEFAULT_MAX_TOKENS).model_dump(),
        stream=False
    )
    
    response = requests.post(url, data=serialize(request), headers={'Content-Type': 'application/json'})
    
    if response.status_code == 200:
        print("RESPONSE:")
        response_data = response.json()
        generated_text = response_data.get("completion_message", {}).get("content", "")
        print(generated_text)
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    completion_stream("Hello, how are you?")
    # completion("What is the capital of France?")
    # chat_completion_stream([{"role": "user", "content": "What's the capital of France?"}])
    # chat_completion([{"role": "user", "content": "What's the weather like today?"}])
