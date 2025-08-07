import os
import logging
import time
import hashlib
import re
import requests
import uvicorn
import gc
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EXPECTED_BEARER_TOKEN = "612aeb3ebe9d63cfdb21e3f7d679fcebde54f7c1283c92b7937ea72c10c966af"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hackerx")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# **MEMORY-OPTIMIZED GLOBAL VARIABLES**
_models = {}  # Lazy-loaded models
_vector_store_cache = {}  # Limited cache
_namespace_timestamps = {}
_cache_limit = 3  # Limit cache to prevent OOM

class HackRxRequest(BaseModel):
    documents: str = Field(..., description="URL to the PDF document")
    questions: List[str] = Field(..., description="List of questions to ask about the document")

class HackRxResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers corresponding to the questions")

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

def cleanup_memory():
    """Force garbage collection to free memory"""
    gc.collect()

def get_model(model_type: str):
    """Lazy load models only when needed to save memory"""
    global _models
    
    if model_type in _models:
        return _models[model_type]
    
    try:
        if model_type == "embeddings":
            logging.info("Loading embeddings model (this may take 30-45 seconds)...")
            # Use lighter model loading with minimal cache
            model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={
                    'device': 'cpu'  # Force CPU to save memory
                },
                encode_kwargs={
                    'batch_size': 8  # Smaller batch for memory
                }
            )
            _models[model_type] = model
            cleanup_memory()
            logging.info("✅ Embeddings model loaded successfully")
            return model
            
        elif model_type == "llm":
            logging.info("Loading LLM model...")
            model = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                temperature=0.0, 
                google_api_key=GOOGLE_API_KEY,
                max_retries=1,  # Reduce retries to save time
                timeout=30  # Correct parameter name
            )
            _models[model_type] = model
            return model
            
        elif model_type == "cross_encoder":
            # Skip cross-encoder to save memory and time
            logging.info("Cross-encoder disabled for memory optimization")
            return None
                
    except Exception as e:
        logging.error(f"Failed to load {model_type} model: {e}")
        # Don't raise exception - try to continue
        return None

def manage_cache():
    """Manage cache size to prevent memory overflow"""
    global _vector_store_cache
    
    if len(_vector_store_cache) >= _cache_limit:
        # Remove oldest cache entry
        oldest_key = min(_vector_store_cache.keys())
        del _vector_store_cache[oldest_key]
        logging.info(f"Removed cache entry: {oldest_key}")
        cleanup_memory()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifespan handler with essential model pre-loading"""
    # Startup
    logging.info("🚀 Starting Ultra-Fast RAG Server...")
    logging.info("🔥 Pre-loading essential models to avoid request timeouts...")
    
    try:
        # Pre-load only the embeddings model (most time-consuming)
        embeddings = get_model("embeddings")
        logging.info("✅ Embeddings model pre-loaded successfully")
        
        # LLM loads quickly, keep it on-demand
        logging.info("📝 LLM will be loaded on-demand")
        
    except Exception as e:
        logging.error(f"❌ Model pre-loading failed: {e}")
        # Continue anyway - models will load on demand
    
    yield
    
    # Shutdown
    logging.info("🔄 Shutting down and cleaning up...")
    global _models, _vector_store_cache
    _models.clear()
    _vector_store_cache.clear()
    cleanup_memory()
    logging.info("✅ Cleanup complete")

# Initialize FastAPI with lifespan
app = FastAPI(
    title="Memory-Optimized RAG System",
    description="Production RAG system optimized for Render's 512MB memory limit",
    version="8.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

async def verify_token(auth_header: str = Security(api_key_header)):
    if auth_header is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization header is missing")
    try:
        scheme, token = auth_header.split()
        if scheme.lower() != "bearer" or token != EXPECTED_BEARER_TOKEN:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid authentication credentials")
    except ValueError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authorization header format")
    return token

def get_document_hash(documents: List) -> str:
    """Generate unique hash for document content"""
    content = "".join([doc.page_content[:100] for doc in documents[:5]])  # Reduced sample size
    return hashlib.md5(content.encode()).hexdigest()[:8]

def detect_document_type(documents: List) -> str:
    """Lightweight document type detection"""
    if not documents:
        return "general"
        
    sample_text = " ".join([doc.page_content[:200] for doc in documents[:2]]).lower()
    
    if any(term in sample_text for term in ["policy", "insurance", "premium"]):
        return "insurance"
    elif any(term in sample_text for term in ["contract", "agreement"]):
        return "legal"
    else:
        return "general"

def memory_efficient_chunking(documents: List, doc_type: str) -> List:
    """Memory-efficient chunking with smaller chunks"""
    chunk_configs = {
        "insurance": {"size": 800, "overlap": 100},  # Smaller chunks
        "legal": {"size": 850, "overlap": 120},
        "general": {"size": 750, "overlap": 80}
    }
    
    config = chunk_configs.get(doc_type, chunk_configs["general"])
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["size"],
        chunk_overlap=config["overlap"],
        separators=["\n\n", "\n", ". ", " "],
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    cleanup_memory()  # Clean up after chunking
    return chunks

def load_document_efficiently(url: str) -> List:
    """Memory-efficient document loading"""
    logging.info(f"Loading document from: {url}")
    
    temp_file_path = None
    try:
        response = requests.get(url, timeout=15, stream=True)  # Stream to save memory
        response.raise_for_status()

        temp_file_path = "temp_doc.pdf"
        
        # Stream write to save memory
        with open(temp_file_path, "wb") as temp_f:
            for chunk in response.iter_content(chunk_size=8192):
                temp_f.write(chunk)

        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()

        # Clean up immediately
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        # Process with memory efficiency
        doc_type = detect_document_type(docs)
        split_docs = memory_efficient_chunking(docs, doc_type)
        
        logging.info(f"Document processed: {len(split_docs)} chunks, type: {doc_type}")
        cleanup_memory()
        
        return split_docs

    except Exception as e:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Document processing failed: {e}")

def monitor_storage_usage(index) -> float:
    """Lightweight storage monitoring"""
    try:
        stats = index.describe_index_stats()
        current_vectors = stats.total_vector_count
        estimated_storage_mb = current_vectors * 0.004
        storage_usage_percent = (estimated_storage_mb / 2000) * 100
        return storage_usage_percent
    except Exception:
        return 0.0

def get_memory_efficient_vector_store(documents: List) -> PineconeVectorStore:
    """Memory-efficient vector store creation"""
    
    start_time = time.time()
    logging.info("Creating memory-efficient vector store...")
    
    try:
        # Check cache first
        doc_hash = get_document_hash(documents)
        cache_key = f"doc_{doc_hash}"
        
        if cache_key in _vector_store_cache:
            logging.info(f"Using cached vector store: {doc_hash}")
            return _vector_store_cache[cache_key]
        
        # Manage cache size
        manage_cache()
        
        # Get models (embeddings should be pre-loaded)
        embeddings = get_model("embeddings")
        if embeddings is None:
            raise HTTPException(status_code=500, detail="Embeddings model failed to load")
        
        # Initialize Pinecone with timeout protection
        pc = Pinecone(api_key=PINECONE_API_KEY)
        existing_indexes = pc.list_indexes().names()
        
        if PINECONE_INDEX_NAME not in existing_indexes:
            alternatives = ["hackerx", "bajajhackerx"]
            index_name = next((alt for alt in alternatives if alt in existing_indexes), 
                            existing_indexes[0] if existing_indexes else None)
            if not index_name:
                raise HTTPException(status_code=404, detail="No suitable index found")
        else:
            index_name = PINECONE_INDEX_NAME
        
        index = pc.Index(index_name)
        
        # Create unique namespace
        namespace = f"doc_{doc_hash}_{int(time.time())}"
        _namespace_timestamps[namespace] = time.time()
        
        # Process documents in smaller batches to save memory
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Smaller batch size for memory efficiency
        batch_size = 16  # Reduced from 32
        vectors_to_upsert = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = embeddings.embed_documents(batch_texts)
            
            # Process this batch immediately
            for j, (text, embedding, metadata) in enumerate(zip(batch_texts, batch_embeddings, metadatas[i:i+batch_size])):
                limited_metadata = {
                    "text": text[:500],  # Further reduced
                    **{k: str(v)[:50] for k, v in metadata.items() if k != "text"}
                }
                
                vectors_to_upsert.append({
                    "id": f"{namespace}_{i+j}",
                    "values": embedding,
                    "metadata": limited_metadata
                })
            
            # Upsert when batch is ready and clear memory
            if len(vectors_to_upsert) >= 50:  # Smaller upsert batches
                index.upsert(vectors=vectors_to_upsert, namespace=namespace)
                vectors_to_upsert.clear()
                cleanup_memory()
                
            logging.info(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        # Upsert remaining vectors
        if vectors_to_upsert:
            index.upsert(vectors=vectors_to_upsert, namespace=namespace)
            cleanup_memory()
        
        # Create vector store
        vector_store = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            namespace=namespace,
            text_key="text"
        )
        
        # Cache with memory management
        _vector_store_cache[cache_key] = vector_store
        
        elapsed_time = time.time() - start_time
        logging.info(f"Memory-efficient vector store created in {elapsed_time:.2f}s")
        
        return vector_store

    except Exception as e:
        elapsed_time = time.time() - start_time
        logging.error(f"Vector store creation failed after {elapsed_time:.2f}s: {e}")
        cleanup_memory()
        raise HTTPException(status_code=500, detail=f"Vector store error: {e}")

def create_lightweight_chain(vector_store: PineconeVectorStore):
    """Create lightweight RAG chain"""
    llm = get_model("llm")
    
    # Simplified prompt to reduce memory usage
    prompt_template = """Answer the question based on the context provided. Be specific and accurate.

Context: {context}
Question: {input}
Answer:"""

    prompt = PromptTemplate.from_template(prompt_template)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(
        vector_store.as_retriever(search_kwargs={"k": 6}),  # Reduced retrieval count
        document_chain
    )

    return retrieval_chain

@app.post("/hackrx/run", response_model=HackRxResponse, dependencies=[Depends(verify_token)])
async def run_submission(request: HackRxRequest):
    """Memory-optimized processing for Render"""
    
    total_start = time.time()
    logging.info("Starting memory-optimized processing...")
    
    try:
        # Load and process document
        documents = load_document_efficiently(request.documents)
        vector_store = get_memory_efficient_vector_store(documents)
        rag_chain = create_lightweight_chain(vector_store)
        
        # Process questions with memory management
        answers = []
        
        for i, question in enumerate(request.questions, 1):
            try:
                response = rag_chain.invoke({"input": question})
                answer = response.get("answer", "Could not generate an answer.")
                answers.append(answer)
                
                # Clean up after each question
                if i % 3 == 0:  # Cleanup every 3 questions
                    cleanup_memory()
                
                logging.info(f"Question {i}/{len(request.questions)} processed")
                
            except Exception as e:
                logging.error(f"Error processing question {i}: {e}")
                answers.append(f"Error: {str(e)}")
        
        total_time = time.time() - total_start
        logging.info(f"Processing completed in {total_time:.2f}s")
        
        # Final cleanup
        cleanup_memory()
        
        return HackRxResponse(answers=answers)

    except HTTPException:
        cleanup_memory()
        raise
    except Exception as e:
        cleanup_memory()
        total_time = time.time() - total_start
        logging.error(f"Processing failed after {total_time:.2f}s: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "message": "Memory-Optimized RAG API for Render", 
        "version": "8.0.0",
        "memory_optimization": "Enabled"
    }

@app.get("/")
async def root():
    return {
        "message": "Memory-Optimized RAG System for Render",
        "version": "8.0.0",
        "optimizations": [
            "Lazy model loading",
            "Memory-efficient chunking",
            "Limited caching (3 entries max)",
            "Aggressive garbage collection",
            "Streaming document loading",
            "Smaller batch sizes"
        ],
        "memory_limit": "512MB (Render compatible)"
    }

@app.get("/memory-status")
async def memory_status():
    """Check memory usage status"""
    import psutil
    import os
    
    try:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        return {
            "memory_usage_mb": round(memory_mb, 2),
            "memory_percent": round(memory_mb / 512 * 100, 2),  # 512MB limit
            "cached_models": list(_models.keys()),
            "cached_vector_stores": len(_vector_store_cache),
            "status": "OK" if memory_mb < 400 else "WARNING" if memory_mb < 480 else "CRITICAL"
        }
    except ImportError:
        return {"error": "psutil not available for memory monitoring"}

@app.post("/clear-memory")
async def clear_memory():
    """Force memory cleanup"""
    global _models, _vector_store_cache
    
    models_cleared = len(_models)
    cache_cleared = len(_vector_store_cache)
    
    _models.clear()
    _vector_store_cache.clear()
    cleanup_memory()
    
    return {
        "message": "Memory cleared",
        "models_cleared": models_cleared,
        "cache_entries_cleared": cache_cleared
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print("🚀 Starting Memory-Optimized RAG Server for Render...")
    print("💾 Memory Limit: 512MB")
    print("🔧 Optimizations: Lazy loading + Limited caching + Garbage collection")
    print(f"🎯 Running on port {port}")
    print("📚 API docs: /docs")
    print("💚 Health check: /health")
    print("📊 Memory status: /memory-status")
    print("🧹 Clear memory: /clear-memory")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        workers=1,  # Single worker to save memory
        reload=False  # Disable reload to save memory
    )