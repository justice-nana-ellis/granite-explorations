import os
import shutil
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
from typing import Optional
import chromadb
import pandas as pd
from docling.document_converter import DocumentConverter

load_dotenv()

# Ollama runs locally — no token needed
# Make sure you have run: ollama pull granite3.3
MODEL_ID = "granite3.3"
EMBED_MODEL = "nomic-embed-text"  # pull with: ollama pull nomic-embed-text

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Ollama doesn't need a real key, but the param is required
)

# ChromaDB stores document chunks locally in ./chroma_db
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("documents")

doc_converter = DocumentConverter()


def embed(text: str) -> list[float]:
    """Generate an embedding for a piece of text using Ollama."""
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding


def chunk_text(text: str, chunk_size: int = 400) -> list[str]:
    """Split text into overlapping chunks so context isn't lost at boundaries."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - 50):  # 50-word overlap
        chunk = " ".join(words[i: i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def parse_file(file_path: str, filename: str) -> list[str]:
    """Parse a file into a list of text chunks based on its type."""
    ext = Path(filename).suffix.lower()
    chunks = []

    if ext in [".pdf", ".docx"]:
        result = doc_converter.convert(file_path)
        text = result.document.export_to_markdown()
        chunks = chunk_text(text)

    elif ext in [".csv"]:
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            text = ", ".join([f"{col}: {val}" for col, val in row.items()])
            chunks.append(text)

    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
        for _, row in df.iterrows():
            text = ", ".join([f"{col}: {val}" for col, val in row.items()])
            chunks.append(text)

    elif ext in [".txt", ".md"]:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_text(text)

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return chunks

app = FastAPI(
    title="Granite Finance API",
    description="A simple API wrapping IBM Granite via Hugging Face Inference API",
    version="1.0.0",
)

SYSTEM_PROMPT = """You are a helpful financial analyst assistant powered by IBM Granite.
Answer questions clearly and precisely. 
When analysing financial data, always cite specific figures and flag any risks you notice.
If you don't know something, say so clearly."""


# --- Request / Response models ---

class ChatRequest(BaseModel):
    message: str
    system_prompt: Optional[str] = None   # override the default system prompt if needed
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7


class ChatResponse(BaseModel):
    reply: str
    model: str
    tokens_used: Optional[int] = None


class MultiTurnRequest(BaseModel):
    messages: list[dict]   # list of {"role": "user"/"assistant"/"system", "content": "..."}
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7


# --- Routes ---

@app.get("/")
def root():
    return {"status": "ok", "model": MODEL_ID}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Single-turn chat endpoint.
    Send a message and get a reply from Granite.
    """
    system = request.system_prompt or SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": request.message},
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Granite API error: {str(e)}")

    reply = response.choices[0].message.content
    tokens = response.usage.total_tokens if response.usage else None

    return ChatResponse(reply=reply, model=MODEL_ID, tokens_used=tokens)


@app.post("/chat/multi-turn", response_model=ChatResponse)
def chat_multi_turn(request: MultiTurnRequest):
    """
    Multi-turn chat endpoint.
    Pass the full conversation history as a list of messages.
    Each message must have 'role' (system/user/assistant) and 'content'.

    Example body:
    {
        "messages": [
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": "What is a P/E ratio?"},
            {"role": "assistant", "content": "A P/E ratio is..."},
            {"role": "user", "content": "How does it compare to P/B ratio?"}
        ]
    }
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages list cannot be empty.")

    # Inject default system prompt if none provided
    has_system = any(m.get("role") == "system" for m in request.messages)
    messages = request.messages if has_system else [
        {"role": "system", "content": SYSTEM_PROMPT},
        *request.messages,
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Granite API error: {str(e)}")

    reply = response.choices[0].message.content
    tokens = response.usage.total_tokens if response.usage else None

    return ChatResponse(reply=reply, model=MODEL_ID, tokens_used=tokens)


@app.post("/analyse")
def analyse(request: ChatRequest):
    """
    Finance-specific analysis endpoint.
    Uses a stricter system prompt focused on financial analysis.
    """
    finance_system = """You are a senior financial analyst.
Analyse the provided data or question with the following rules:
- Be precise with numbers and percentages
- Always mention key risks
- Structure your output with clear sections
- Flag any missing information that would improve the analysis"""

    messages = [
        {"role": "system", "content": finance_system},
        {"role": "user",   "content": request.message},
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=request.max_tokens or 768,
            temperature=0.3,  # lower temperature for factual analysis
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Granite API error: {str(e)}")

    return {"analysis": response.choices[0].message.content, "model": MODEL_ID}


# --- Document / RAG endpoints ---

class RAGRequest(BaseModel):
    question: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.3
    num_sources: Optional[int] = 4  # how many document chunks to retrieve


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF, Word, Excel, CSV, TXT) to be indexed.
    The file is parsed, split into chunks, embedded, and stored in ChromaDB.
    """
    allowed = {".pdf", ".docx", ".xlsx", ".xls", ".csv", ".txt", ".md"}
    ext = Path(file.filename).suffix.lower()

    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed)}"
        )

    # Save the uploaded file temporarily so docling/pandas can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        chunks = parse_file(tmp_path, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse file: {str(e)}")
    finally:
        os.unlink(tmp_path)  # always clean up the temp file

    if not chunks:
        raise HTTPException(status_code=400, detail="No text could be extracted from the file.")

    # Embed each chunk and store in ChromaDB
    try:
        embeddings = [embed(chunk) for chunk in chunks]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding error (is nomic-embed-text pulled?): {str(e)}")

    # Use filename + index as unique IDs so re-uploading the same file overwrites it
    ids = [f"{file.filename}::{i}" for i in range(len(chunks))]
    metadatas = [{"source": file.filename, "chunk": i} for i in range(len(chunks))]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return {
        "message": f"'{file.filename}' indexed successfully.",
        "chunks_stored": len(chunks),
    }


@app.get("/documents")
def list_documents():
    """List all documents that have been indexed."""
    results = collection.get(include=["metadatas"])
    if not results["metadatas"]:
        return {"documents": [], "total_chunks": 0}

    # Deduplicate to just unique filenames
    sources = list({m["source"] for m in results["metadatas"]})
    return {"documents": sources, "total_chunks": len(results["metadatas"])}


@app.delete("/documents")
def clear_documents():
    """Delete all indexed documents from the vector store."""
    global collection
    chroma_client.delete_collection("documents")
    collection = chroma_client.get_or_create_collection("documents")
    return {"message": "All documents cleared."}


@app.post("/chat/rag")
def chat_rag(request: RAGRequest):
    """
    Ask a question and get an answer grounded in your uploaded documents.
    Finds the most relevant chunks from your documents and sends them to Granite as context.
    """
    total = collection.count()
    if total == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Upload files via POST /documents/upload first."
        )

    # Embed the question and find the most relevant document chunks
    try:
        query_embedding = embed(request.question)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding error: {str(e)}")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(request.num_sources, total),
        include=["documents", "metadatas"],
    )

    context_chunks = results["documents"][0]
    sources = list({m["source"] for m in results["metadatas"][0]})
    context = "\n\n---\n\n".join(context_chunks)

    messages = [
        {
            "role": "system",
            "content": """You are a financial analyst assistant.
Answer questions using ONLY the document context provided below.
Always cite which document your answer comes from.
If the answer is not in the context, say 'I could not find that in the uploaded documents.'
Be precise with numbers and figures.""",
        },
        {
            "role": "user",
            "content": f"Document context:\n\n{context}\n\nQuestion: {request.question}",
        },
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Granite API error: {str(e)}")

    return {
        "answer": response.choices[0].message.content,
        "sources": sources,
        "model": MODEL_ID,
    }
