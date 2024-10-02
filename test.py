import requests

def test_inference():
    url = "http://localhost:8000/generate-stream"
    test_text = "I love"
    
    response = requests.post(url, json={"text": test_text}, stream=True)
    
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

if __name__ == "__main__":
    test_inference()