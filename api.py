from fastapi import FastAPI, UploadFile, File
import requests
import docx

app = FastAPI()

@app.get("/")
def home():
    return {"message": "AI Paper Summarizer"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"
        HEADERS = {"Authorization": "Bearer YOUR_HF_TOKEN"}

        # Handle file types
        if file.filename.endswith(".txt"):
            content = await file.read()
            text = content.decode("utf-8")

        elif file.filename.endswith(".docx"):
            doc = docx.Document(file.file)
            text = "\n".join([para.text for para in doc.paragraphs])

        else:
            return {"error": "Only .txt and .docx supported"}

        response = requests.post(API_URL, headers=HEADERS, json={"inputs": text})

        return response.json()

    except Exception as e:
        return {"error": str(e)}
