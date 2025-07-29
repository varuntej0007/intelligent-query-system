import os
import requests
from fastapi import FastAPI, Request, HTTPException, Security
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

# --- Configuration ---
# Load environment variables from a .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- API Key Authentication ---
API_KEY = "cf9b8191780fa01632ebe2dbe4978af143231eb6d8efb49f0a727fe116f10143"
API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    """
    Dependency to validate the API key.
    The key is expected in the format "Bearer <key>".
    """
    if not api_key_header.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Invalid authorization scheme.")
    
    token = api_key_header.split(" ")[1]
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    return token

# --- FastAPI App Initialization ---
app = FastAPI(
    title="LLM-Powered Intelligent Query–Retrieval System",
    description="Processes documents to answer natural language questions.",
    version="1.0.0"
)

# --- Pydantic Models for API ---
class QueryRequest(BaseModel):
    documents: str = Field(..., description="URL to the PDF document.")
    questions: List[str] = Field(..., description="List of questions to answer based on the document.")

class Answer(BaseModel):
    question: str
    answer: str
    source_text: Optional[str] = None
    
class QueryResponse(BaseModel):
    answers: List[str]

# --- Global Variables & Models ---
# Using a thread pool for concurrent network and CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=os.cpu_count())

# Load the sentence transformer model once at startup
# Switching back to the more accurate model for better performance.
logger.info("Loading Sentence Transformer model...")
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Sentence Transformer model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Sentence Transformer model: {e}")
    model = None

# --- Core Logic: Document Processing and Q&A ---

def download_and_extract_text(pdf_url: str) -> str:
    """
    Downloads a PDF from a URL and extracts text from it.
    """
    logger.info(f"Downloading document from: {pdf_url}")
    try:
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        logger.info("Document downloaded. Extracting text.")
        with BytesIO(response.content) as f:
            reader = PdfReader(f)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        logger.info(f"Text extracted. Total characters: {len(text)}")
        return text
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download PDF: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download PDF from URL: {e}")
    except Exception as e:
        logger.error(f"Failed to process PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF file: {e}")

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Splits a long text into smaller, overlapping chunks.
    This helps in creating more focused embeddings.
    """
    logger.info("Chunking extracted text.")
    if not text:
        return []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    logger.info(f"Text chunked into {len(chunks)} chunks.")
    return chunks

def create_faiss_index(chunks: List[str], model: SentenceTransformer) -> Optional[faiss.Index]:
    """
    Creates a FAISS index for a list of text chunks.
    FAISS allows for very fast similarity searches.
    """
    if not chunks or model is None:
        logger.warning("No chunks or model available to create FAISS index.")
        return None
        
    logger.info("Creating embeddings for text chunks...")
    try:
        embeddings = model.encode(chunks, convert_to_tensor=False, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32') # FAISS requires float32
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)  # Using L2 distance for similarity
        index.add(embeddings)
        
        logger.info(f"FAISS index created successfully with {index.ntotal} vectors.")
        return index
    except Exception as e:
        logger.error(f"Error creating FAISS index: {e}")
        return None

def search_faiss_index(query: str, index: faiss.Index, chunks: List[str], model: SentenceTransformer, k: int = 5) -> List[str]:
    """
    Searches the FAISS index for the most relevant chunks for a given query.
    """
    if index is None or model is None:
        return []
        
    logger.info(f"Searching for relevant chunks for query: '{query[:50]}...'")
    query_embedding = model.encode([query], convert_to_tensor=False)
    query_embedding = np.array(query_embedding).astype('float32')
    
    distances, indices = index.search(query_embedding, k)
    
    # Return the actual text chunks corresponding to the top indices
    return [chunks[i] for i in indices[0] if i < len(chunks)]

async def generate_answer_with_llm(question: str, context_chunks: List[str]) -> str:
    """
    Uses the Gemini API to generate an answer based on the question and retrieved context.
    """
    # Get the API key from environment variables
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        error_msg = "GEMINI_API_KEY environment variable not found. Please set it to your Google AI Studio API key."
        logger.error(error_msg)
        return f"Configuration Error: {error_msg}"

    if not context_chunks:
        return "I could not find relevant information in the document to answer this question."

    context_separator = "\n---\n"
    prompt = f"""
    You are an expert Q&A system for policy documents.
    Based *only* on the following context, please provide a precise and concise answer to the question.
    If the context contains the direct answer, quote it.
    If the information is not in the context, state that the answer is not found in the provided text.
    Do not use any external knowledge.

    Context:
    ---
    {context_separator.join(context_chunks)}
    ---

    Question: {question}

    Precise Answer:
    """
    
    logger.info(f"Generating answer for: '{question}'")
    
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={gemini_api_key}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "topP": 0.95,
            "maxOutputTokens": 500,
        }
    }

    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            executor,
            lambda: requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'})
        )
        response.raise_for_status()
        result = response.json()
        
        if (candidates := result.get("candidates")) and \
           (content := candidates[0].get("content")) and \
           (parts := content.get("parts")):
            answer_text = parts[0].get("text", "No answer could be generated.").strip()
            logger.info(f"Successfully generated answer for: '{question}'")
            return answer_text
        else:
            logger.warning(f"Could not extract answer from LLM response for question: '{question}'. Response: {result}")
            return "Error: The model did not provide a valid answer."

    except requests.exceptions.RequestException as e:
        logger.error(f"API call to LLM failed: {e}")
        # Provide a more specific error message if it's a 4xx error
        if e.response is not None and 400 <= e.response.status_code < 500:
             return f"Error: API request failed with status {e.response.status_code}. This could be due to an invalid API key or billing issues. Please check your Google AI Studio account."
        return f"Error: Could not connect to the language model API. {e}"
    except (KeyError, IndexError) as e:
        logger.error(f"Invalid response structure from LLM API: {e}. Response: {result}")
        return "Error: Received an invalid response from the language model."
    except Exception as e:
        logger.error(f"An unexpected error occurred during LLM answer generation: {e}")
        return "An unexpected error occurred while generating the answer."


# --- API Endpoint Implementation ---
@app.post("/hackrx/run", response_model=QueryResponse, tags=["Query System"])
async def run_submission(request: QueryRequest, api_key: str = Security(get_api_key)):
    """
    This endpoint orchestrates the entire query-retrieval process:
    1.  Downloads and parses the document from the provided URL.
    2.  Chunks the text for processing.
    3.  Creates embeddings and a FAISS index for semantic search.
    4.  For each question:
        a. Searches the index for relevant context.
        b. Prompts an LLM with the context and question to get an answer.
    5.  Returns a structured JSON response with the answers.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available. The service is starting up or encountered an error.")

    loop = asyncio.get_running_loop()

    try:
        document_text = await loop.run_in_executor(executor, download_and_extract_text, request.documents)
        text_chunks = await loop.run_in_executor(executor, chunk_text, document_text)
        if not text_chunks:
            raise HTTPException(status_code=400, detail="Could not extract any text from the document.")
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error during document processing: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during document processing.")

    faiss_index = await loop.run_in_executor(executor, create_faiss_index, text_chunks, model)
    if faiss_index is None:
        raise HTTPException(status_code=500, detail="Failed to create the search index for the document.")

    async def process_question(question: str):
        relevant_chunks = await loop.run_in_executor(
            executor, search_faiss_index, question, faiss_index, text_chunks, model
        )
        answer = await generate_answer_with_llm(question, relevant_chunks)
        return answer

    tasks = [process_question(q) for q in request.questions]
    
    logger.info(f"Processing {len(request.questions)} questions in parallel.")
    answers_list = await asyncio.gather(*tasks, return_exceptions=True)
    
    final_answers = []
    for i, result in enumerate(answers_list):
        if isinstance(result, Exception):
            logger.error(f"Error processing question '{request.questions[i]}': {result}")
            final_answers.append(f"An error occurred while answering this question: {result}")
        else:
            final_answers.append(result)

    return QueryResponse(answers=final_answers)

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Welcome to the Intelligent Query–Retrieval System. See /docs for API documentation."}

# To run the app, use the command:
# uvicorn main:app --reload
