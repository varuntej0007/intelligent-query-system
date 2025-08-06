# main.py - High Accuracy Version (Target: >50%)

import os
import requests
from fastapi import FastAPI, Request, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader
from io import BytesIO
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from dotenv import load_dotenv
import re
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="High Accuracy LLM Query System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("API_KEY", "cf9b8191780fa01632ebe2dbe4978af143231eb6d8efb49f0a727fe116f10143")
API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if not api_key_header.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Invalid authorization scheme.")
    token = api_key_header.split(" ")[1]
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid credentials.")
    return token

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

executor = ThreadPoolExecutor(max_workers=os.cpu_count())
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Helper Functions ---
def download_and_extract_text(pdf_url: str) -> str:
    response = requests.get(pdf_url, timeout=30)
    response.raise_for_status()
    reader = PdfReader(BytesIO(response.content))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += re.sub(r'\s*\n\s*', ' ', page_text.strip()) + "\n"
    return text

def chunk_text(text: str) -> List[str]:
    paragraphs = re.split(r'\n\s*\n', text)
    chunks, buffer = [], ""
    for p in paragraphs:
        buffer += p + "\n"
        if len(buffer) > 1000:
            chunks.append(buffer.strip())
            buffer = ""
    if buffer:
        chunks.append(buffer.strip())
    return [c for c in chunks if len(c) > 50]

def create_faiss_index(chunks: List[str]) -> faiss.Index:
    embeddings = model.encode(chunks, convert_to_tensor=False, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def search_index(query: str, index: faiss.Index, chunks: List[str], k: int = 12) -> List[str]:
    query_vec = model.encode([query], convert_to_tensor=False, normalize_embeddings=True).astype('float32')
    distances, indices = index.search(query_vec, k)
    results = [chunks[i] for i, score in zip(indices[0], distances[0]) if i < len(chunks) and score > 0.6]
    return results[:6]  # Cap context size to 6 chunks for better focus

async def call_gemini(prompt: str) -> str:
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        return "GEMINI_API_KEY not configured."
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={gemini_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "topP": 0.9,
            "maxOutputTokens": 300
        }
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception as e:
        return f"LLM error: {str(e)}"

# --- Core Endpoint ---
@app.post("/hackrx/run", response_model=QueryResponse)
async def run_query(request: QueryRequest, api_key: str = Security(get_api_key)):
    loop = asyncio.get_running_loop()
    text = await loop.run_in_executor(executor, download_and_extract_text, request.documents)
    chunks = await loop.run_in_executor(executor, chunk_text, text)
    index = await loop.run_in_executor(executor, create_faiss_index, chunks)

    async def process(q):
        context = await loop.run_in_executor(executor, search_index, q, index, chunks)
        prompt = f"""
You are a document expert AI.
Answer the following question using ONLY the information in the context.
If no direct answer exists, reply with: 'Answer not found in the provided context.'

Context:
---\n{chr(10).join(context)}\n---
Question: {q}
Step-by-step reasoning followed by final answer:
"""
        return await call_gemini(prompt)

    answers = await asyncio.gather(*[process(q) for q in request.questions])
    return QueryResponse(answers=answers)

@app.get("/")
async def root():
    return {"message": "High Accuracy API is running. Visit /docs"}
