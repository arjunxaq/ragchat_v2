from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uuid
from typing import List, Dict
import PyPDF2
import io
import requests
import json
import numpy as np

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Together AI configuration
TOGETHER_API_KEY = "TogetherAPIKEY"
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

# In-memory storage for documents
document_store: Dict[str, Dict] = {}

class ChatRequest(BaseModel):
    question: str
    collection_id: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF file"""
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into chunks with overlap"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks

def create_embedding(text: str) -> np.ndarray:
    """Create a simple TF-IDF style embedding"""
    # Normalize text
    text_lower = text.lower()
    words = text_lower.split()
    
    # Create word frequency vector
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Convert to array (use hash for fixed size)
    embedding = np.zeros(256)
    for word, freq in word_freq.items():
        idx = hash(word) % 256
        embedding[idx] += freq
    
    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

def retrieve_relevant_chunks(query: str, chunks: List[str], top_k: int = 3) -> List[str]:
    """Retrieve most relevant chunks for a query"""
    query_embedding = create_embedding(query)
    
    # Calculate similarity for each chunk
    similarities = []
    for chunk in chunks:
        chunk_embedding = create_embedding(chunk)
        sim = cosine_similarity(query_embedding, chunk_embedding)
        similarities.append(sim)
    
    # Get top-k chunks
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def query_together_ai(prompt: str, context: str) -> str:
    """Query Together AI API"""
    if not TOGETHER_API_KEY:
        return "Error: TOGETHER_API_KEY not set. Please set your Together AI API key."
    
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the provided context. If the answer is not in the context, say so politely."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {prompt}\n\nPlease provide a clear and concise answer based on the context above."
            }
        ],
        "max_tokens": 512,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(TOGETHER_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Error querying Together AI: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

@app.get("/")
def read_root():
    return {"message": "RAGChat Backend API", "status": "running"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process PDF file"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Read PDF
        pdf_content = await file.read()
        text = extract_text_from_pdf(pdf_content)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text found in PDF")
        
        # Create chunks
        chunks = chunk_text(text)
        
        # Store in memory
        collection_id = str(uuid.uuid4())
        document_store[collection_id] = {
            "filename": file.filename,
            "chunks": chunks,
            "text": text
        }
        
        return {
            "collection_id": collection_id,
            "filename": file.filename,
            "chunks": len(chunks)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the PDF content"""
    if request.collection_id not in document_store:
        raise HTTPException(status_code=404, detail="Collection not found. Please upload a PDF first.")
    
    try:
        doc = document_store[request.collection_id]
        chunks = doc["chunks"]
        
        # Retrieve relevant chunks
        relevant_chunks = retrieve_relevant_chunks(request.question, chunks, top_k=3)
        
        if not relevant_chunks:
            return ChatResponse(
                answer="I couldn't find relevant information in the document to answer your question.",
                sources=[]
            )
        
        # Create context from retrieved documents
        context = "\n\n".join(relevant_chunks)
        
        # Query Together AI
        answer = query_together_ai(request.question, context)
        
        return ChatResponse(
            answer=answer,
            sources=relevant_chunks
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "collections": len(document_store),
        "api_key_set": bool(TOGETHER_API_KEY)
    }

@app.delete("/collection/{collection_id}")
def delete_collection(collection_id: str):
    """Delete a collection from memory"""
    if collection_id in document_store:
        del document_store[collection_id]
        return {"message": "Collection deleted", "collection_id": collection_id}
    raise HTTPException(status_code=404, detail="Collection not found")