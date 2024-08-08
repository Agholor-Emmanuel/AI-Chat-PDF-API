import document_processor
import embeddings
import model
import os
import time
from pypdf import PdfReader
from typing import Optional
from fastapi import FastAPI, UploadFile, Request
from pydantic import BaseModel
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session_state = None

class UserInput(BaseModel):
    user_input: str

def process_pdf(filename):
    # Process the PDF from local file
    pdf_source = f'uploads/{filename}'
    file = open(pdf_source, 'rb')

    # Extract text from PDF
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    file.close()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks

def compute(raw_text, model_name):
    print("Training model on your document(s)")
    text_chunks = get_text_chunks(raw_text)
    vector_database = embeddings.get_vector_database(text_chunks, model_name)
    return vector_database


@app.get("/")
async def root():
    return "Welcome to Doc parser"

@app.post("/uploadfile/")
async def upload_file(file: UploadFile):
    try:
        contents = await file.read()
        path = os.path.join('uploads/', file.filename)
        with open(path, 'wb') as f:
            f.write(contents)
        raw_text = process_pdf(file.filename)
        model_name = 'Mistral7B'
        vector_db = compute(raw_text, model_name)
        global session_state 
        session_state = model.chatbot(vector_db, model_name)
        print(session_state)
        result = {"filename": file.filename, "success": True}
    except Exception:
        result = {"detail": "Please upload a proper document that hasn't been scanned and isn't a picture.",
                  "filename": f"{file.filename} couldn't be parsed",
                  "success": False}
    return result

@app.post("/chat/")
async def conversation(input: UserInput):
    try:
        print(input.user_input)
        print("================================================================================")
        start = time.time()
        response = session_state({'question': input.user_input})
        end = time.time()
        print(f'Time taken to run {end-start}secs')
    except Exception:
        response = {"detail":"Please Upload a Document"}
    return response

