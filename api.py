from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os

app = FastAPI(title="Brain Tumor SLM Summarizer API")

# Setup Paths
MODEL_DIR = os.getenv("MODEL_DIR", "./models/brain-tumor-slm")
tokenizer = None
model = None

class SummarizeRequest(BaseModel):
    text: str

class SummarizeResponse(BaseModel):
    summary: str

@app.on_event("startup")
def load_model():
    global tokenizer, model
    try:
        print(f"Loading model from {MODEL_DIR}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load model from {MODEL_DIR}. You must train the model first or map the volume. Error: {e}")

@app.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train the model first.")
        
    inputs = tokenizer(request.text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=150, min_length=30, do_sample=False)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return SummarizeResponse(summary=summary)

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}
