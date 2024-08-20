from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel

app = FastAPI()
pipe = pipeline("text-generation", model="openai-community/gpt2")

class TextGenerationRequest(BaseModel):
    prompt: str
    max_length: int = 1000

@app.post("/generate_text")
async def generate_text(request: TextGenerationRequest):
    result = pipe(request.prompt, max_length=request.max_length, num_return_sequences=1, truncation=True)
    return {"generated_text": result[0]["generated_text"]}

@app.get("/")
async def root():
    return {"message": "Hello World"}
