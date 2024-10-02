import requests
from constants import Config

def generate_stream(text):
    url = f"http://localhost:{Config.PORT}/generate-stream"
    
    response = requests.post(url, json={"text": text}, stream=True)
    
    if response.status_code == 200:
        print("RESPONSE STREAM:")
        total_response = ""
        for i, chunk in enumerate(response.iter_content(decode_unicode=True)):
            if chunk:
                print(f"{i+1}: {chunk}")
                total_response += chunk
        print("\nTOTAL RESPONSE:")
        print(total_response)
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

def generate(text):
    url = f"http://localhost:{Config.PORT}/generate"
    
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
    # generate_stream()
    generate("I love")
