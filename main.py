from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import requests
from docx import Document
import io
import os

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your Vercel URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "AI Paper Summarizer running 🚀"}


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        API_URL = "https://router.huggingface.co/hf-inference/models/google/flan-t5-small"

        HF_TOKEN = os.getenv("HF_TOKEN")

        
        if not HF_TOKEN:
            return {"error": "HF_TOKEN not set in environment variables"}

        HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

        content = await file.read()

        
        if len(content) > 1_000_000:
            return {"error": "File too large (max 1MB)"}

        
        if file.filename.endswith(".txt"):
            text = content.decode("utf-8")

        elif file.filename.endswith(".docx"):
            doc = Document(io.BytesIO(content))
            text = "\n".join([para.text for para in doc.paragraphs])

        else:
            return {"error": "Only .txt and .docx supported"}

       
        text = text[:2000]

        
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json={"inputs": f"Summarize this:\n{text}"},
            timeout=30
        )

        result = response.json()

        
        if isinstance(result, list) and "generated_text" in result[0]:
            return {"summary": result[0]["generated_text"]}

        return {"raw_response": result}

    except Exception as e:
        return {"error": str(e)}
