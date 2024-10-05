import requests
from config import Config
import time

default_url = f"http://localhost:{Config.PORT}"

def generate_stream(text):
    url = f"{default_url}/generate-stream"
    
    response = requests.post(url, json={"text": text}, stream=True)
    
    if response.status_code == 200:
        print("RESPONSE STREAM:")
        total_response = ""
        start_time = time.time()
        token_count = 0
        for i, chunk in enumerate(response.iter_content(decode_unicode=True)):
            if chunk:
                print(f"{i+1}: {chunk}")
                total_response += chunk
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

def generate(text):
    url = f"{default_url}/generate"
    
    response = requests.post(url, json={"text": text})
    
    if response.status_code == 200:
        print("RESPONSE:")
        response_data = response.json()
        generated_text = response_data.get("text", "")
        print(generated_text)
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    generate_stream("hello there")
    # generate("I love")
