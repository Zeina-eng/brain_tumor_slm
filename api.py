from fastapi import FastAPI, UploadFile, File
import requests

app = FastAPI()

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"
HEADERS = {"Authorization": "Bearer YOUR_HF_TOKEN"}

@app.get("/")
def home():
    return {"message": "AI Paper Summarizer"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")

    response = requests.post(API_URL, headers=HEADERS, json={"inputs": text})

    return response.json()
