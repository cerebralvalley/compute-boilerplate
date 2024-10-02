from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn
from constants import Config
import asyncio

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(Config.MODEL_NAME).to(device)

@app.post("/generate-stream")
async def generate_text(request: Request):
    data = await request.json()
    input_text = data.get("text", "")

    async def generate():
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)  # tensor of shape (1, input_text_length) of token IDs
        attention_mask = torch.ones_like(input_ids).to(device)  # initializing attention mask to be all ones (attend to all tokens)
        past = None  # the attention key/value cache, used to speed up attention calculation, will be populated as we inference
        
        for _ in range(Config.MAX_TOKENS):
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past, use_cache=True)
                token = outputs.logits[:, -1, :].argmax(dim=-1).to(device)  # selecting the most likely next token from the (presumably softmax) output logics
                
                token_str = tokenizer.decode(token.item())
                yield token_str
                await asyncio.sleep(0.1)
                
                input_ids = token.unsqueeze(0)  # for one token, reassign input_ids to be tensor of shape (1, 1) via unsqueeze (dimension add) function
                attention_mask = torch.cat([attention_mask, torch.ones_like(input_ids)], dim=-1)  # extend attention mask to include the new token
                past = outputs.past_key_values

    return StreamingResponse(generate(), media_type="text/plain")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=Config.PORT)
