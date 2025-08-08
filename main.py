import os
import logging
import time
import hashlib
import gc
import requests
import uvicorn
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# --- RENDER-OPTIMIZED CONFIG ---
load_dotenv()
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # Ensure logs go to stdout for Render
)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EXPECTED_BEARER_TOKEN = os.getenv("BEARER_TOKEN", "612aeb3ebe9d63cfdb21e3f7d679fcebde54f7c1283c92b7937ea72c10c966af")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hackerx")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- AGGRESSIVE MEMORY CONTROL ---
_models = {}
_vector_store_cache = {}
_namespace_cache = {}
_cache_limit = 2  # Reduced from 3 to save memory
_max_chunks_per_doc = 100  # Limit document size

def aggressive_cleanup():
    """Aggressive memory cleanup for Render"""
    global _models, _vector_store_cache
    # Clear temporary variables
    for obj in gc.get_objects():
        if hasattr(obj, 'clear') and callable(obj.clear):
            try:
                obj.clear()
            except:
                pass
    gc.collect()
    gc.collect()  # Double cleanup for Render

def monitor_memory():
    """Monitor memory usage and force cleanup if needed"""
    try:
        import psutil
        mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        if mem_mb > 450:  # Approaching 512MB limit
            logging.warning(f"High memory usage: {mem_mb:.1f}MB - forcing cleanup")
            aggressive_cleanup()
        return mem_mb
    except ImportError:
        return 0

# --- OPTIMIZED MODEL LOADING ---
def get_model_safe(model_type: str):
    """Safe model loading with memory monitoring"""
    if model_type in _models:
        return _models[model_type]
    
    # Check memory before loading
    monitor_memory()
    
    try:
        if model_type == "embeddings":
            logging.info("Loading lightweight embeddings model...")
            model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={
                    "device": "cpu",
                    "normalize_embeddings": True  # Better for similarity search
                },
                encode_kwargs={
                    "batch_size": 4,  # Smaller batch size for Render
                    "convert_to_tensor": False,  # Save memory
                    "normalize_embeddings": True
                }
            )
            _models[model_type] = model
            aggressive_cleanup()
            logging.info("✅ Embeddings model loaded")
            return model
            
        elif model_type == "llm":
            logging.info("Loading Gemini LLM...")
            model = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.0,
                google_api_key=GOOGLE_API_KEY,
                max_retries=1,
                timeout=25,  # Shorter timeout for Render
                max_output_tokens=1024  # Limit response length
            )
            _models[model_type] = model
            logging.info("✅ LLM loaded")
            return model
            
    except Exception as e:
        logging.error(f"Failed to load {model_type}: {e}")
        # Don't crash - return None and handle gracefully
        return None

def smart_cache_management():
    """Intelligent cache management for Render"""
    global _vector_store_cache, _namespace_cache
    
    if len(_vector_store_cache) >= _cache_limit:
        # Remove oldest entries
        to_remove = len(_vector_store_cache) - _cache_limit + 1
        for _ in range(to_remove):
            if _vector_store_cache:
                oldest_key = next(iter(_vector_store_cache))
                del _vector_store_cache[oldest_key]
                if oldest_key in _namespace_cache:
                    del _namespace_cache[oldest_key]
        aggressive_cleanup()
        logging.info(f"Cache cleaned, {len(_vector_store_cache)} entries remaining")

# --- RENDER-OPTIMIZED LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Optimized startup for Render"""
    logging.info("🚀 Starting Render-Optimized RAG System...")
    
    # Pre-load only embeddings (most critical)
    try:
        embeddings_model = get_model_safe("embeddings")
        if embeddings_model:
            logging.info("✅ Essential models pre-loaded")
        else:
            logging.warning("⚠️ Embeddings pre-load failed - will load on demand")
    except Exception as e:
        logging.error(f"Startup error: {e} - continuing with on-demand loading")
    
    # Monitor initial memory
    mem = monitor_memory()
    logging.info(f"📊 Startup memory usage: {mem:.1f}MB")
    
    yield
    
    # Cleanup on shutdown
    logging.info("🔄 Shutting down...")
    _models.clear()
    _vector_store_cache.clear()
    _namespace_cache.clear()
    aggressive_cleanup()
    logging.info("✅ Cleanup complete")

# --- FASTAPI APP ---
app = FastAPI(
    title="Render-Optimized RAG System",
    description="Memory-optimized RAG system designed for Render's 512MB limit",
    version="10.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

async def verify_token(auth_header: str = Security(api_key_header)):
    if not auth_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Authorization header missing"
        )
    try:
        scheme, token = auth_header.split(maxsplit=1)
        if scheme.lower() != "bearer" or token != EXPECTED_BEARER_TOKEN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, 
                detail="Invalid credentials"
            )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Invalid authorization format"
        )
    return token

# --- DATA MODELS ---
class HackRxRequest(BaseModel):
    documents: str = Field(..., description="URL to the PDF document")
    questions: List[str] = Field(..., description="List of questions")

class HackRxResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers")

# --- MEMORY-EFFICIENT DOCUMENT PROCESSING ---
def get_document_hash(documents: List) -> str:
    """Generate hash for caching"""
    if not documents:
        return "empty"
    # Use smaller sample for hash to save memory
    content = "".join([doc.page_content[:50] for doc in documents[:3]])
    return hashlib.md5(content.encode()).hexdigest()[:8]

def render_optimized_document_processing(url: str) -> List:
    """Memory-efficient document processing for Render"""
    temp_file = None
    try:
        logging.info(f"📄 Downloading: {url}")
        
        # Stream download to save memory
        response = requests.get(url, timeout=20, stream=True)
        response.raise_for_status()
        
        temp_file = f"temp_{int(time.time())}.pdf"
        with open(temp_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=4096):  # Smaller chunks
                if chunk:
                    f.write(chunk)
        
        # Load and process document
        loader = PyPDFLoader(temp_file)
        docs = loader.load()
        
        # Clean up temp file immediately
        os.remove(temp_file)
        temp_file = None
        
        if not docs:
            raise HTTPException(status_code=400, detail="No content extracted from PDF")
        
        # Optimized chunking for Render
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Smaller chunks for memory efficiency
            chunk_overlap=100,  # Reduced overlap
            separators=["\n\n", "\n", ". ", ".", " "],
            length_function=len
        )
        
        chunks = splitter.split_documents(docs)
        
        # Limit chunks to prevent OOM
        if len(chunks) > _max_chunks_per_doc:
            logging.warning(f"Limiting chunks from {len(chunks)} to {_max_chunks_per_doc}")
            chunks = chunks[:_max_chunks_per_doc]
        
        logging.info(f"✅ Created {len(chunks)} chunks")
        
        # Cleanup before returning
        del docs, response
        aggressive_cleanup()
        
        return chunks
        
    except Exception as e:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)
        logging.error(f"Document processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

# --- MEMORY-EFFICIENT VECTOR STORE ---
def create_memory_efficient_vector_store(documents: List) -> PineconeVectorStore:
    """Create vector store optimized for Render's memory constraints"""
    start_time = time.time()
    
    try:
        # Check cache first
        doc_hash = get_document_hash(documents)
        cache_key = f"vs_{doc_hash}"
        
        if cache_key in _vector_store_cache:
            logging.info(f"📦 Using cached vector store: {doc_hash}")
            return _vector_store_cache[cache_key]
        
        # Manage cache before creating new store
        smart_cache_management()
        
        # Get embeddings model
        embeddings = get_model_safe("embeddings")
        if not embeddings:
            raise HTTPException(status_code=500, detail="Embeddings model failed to load")
        
        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        indexes = pc.list_indexes().names()
        
        # Select index
        if PINECONE_INDEX_NAME in indexes:
            index_name = PINECONE_INDEX_NAME
        elif "hackerx" in indexes:
            index_name = "hackerx"
        elif "bajajhackerx" in indexes:
            index_name = "bajajhackerx"
        else:
            index_name = indexes[0] if indexes else None
            
        if not index_name:
            raise HTTPException(status_code=404, detail="No suitable Pinecone index found")
        
        # Create unique namespace
        namespace = f"doc_{doc_hash}_{int(time.time())}"
        
        # Process documents in small batches to avoid memory spikes
        logging.info("🔄 Processing documents in memory-efficient batches...")
        
        texts = [doc.page_content for doc in documents]
        metadatas = [{"text": doc.page_content[:300]} for doc in documents]  # Limit metadata size
        
        # Create vector store with batch processing
        vector_store = PineconeVectorStore.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            index_name=index_name,
            namespace=namespace
        )
        
        # Cache the result
        _vector_store_cache[cache_key] = vector_store
        _namespace_cache[cache_key] = namespace
        
        # Cleanup
        del texts, metadatas
        aggressive_cleanup()
        
        elapsed = time.time() - start_time
        logging.info(f"✅ Vector store created in {elapsed:.1f}s")
        
        return vector_store
        
    except Exception as e:
        aggressive_cleanup()
        logging.error(f"Vector store creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Vector store creation failed: {str(e)}")

# --- LIGHTWEIGHT RAG CHAIN ---
def create_render_optimized_chain(vector_store: PineconeVectorStore):
    """Create memory-efficient RAG chain"""
    llm = get_model_safe("llm")
    if not llm:
        raise HTTPException(status_code=500, detail="LLM failed to load")
    
    # Simplified prompt to reduce memory usage
    prompt_template = """Answer the question based on the provided context. Be specific and accurate.

Context: {context}

Question: {input}

Answer:"""
    
    prompt = PromptTemplate.from_template(prompt_template)
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Use smaller retrieval count for memory efficiency
    retrieval_chain = create_retrieval_chain(
        vector_store.as_retriever(search_kwargs={"k": 6}),  # Reduced from 10
        document_chain
    )
    
    return retrieval_chain

# --- OPTIMIZED QUESTION PROCESSING ---
async def process_questions_efficiently(questions: List[str], rag_chain) -> List[str]:
    """Process questions with memory monitoring"""
    answers = []
    
    for i, question in enumerate(questions, 1):
        try:
            # Monitor memory before each question
            mem_usage = monitor_memory()
            
            if mem_usage > 480:  # Critical memory usage
                logging.warning(f"Critical memory usage: {mem_usage:.1f}MB - forcing cleanup")
                aggressive_cleanup()
            
            # Process question
            result = rag_chain.invoke({"input": question.strip()})
            answer = result.get("answer", "Information not found in the provided context.")
            answers.append(answer)
            
            # Cleanup every 2 questions on Render
            if i % 2 == 0:
                aggressive_cleanup()
            
            logging.info(f"✅ Question {i}/{len(questions)} processed")
            
        except Exception as e:
            logging.error(f"❌ Question {i} failed: {e}")
            answers.append(f"Error processing question: {str(e)}")
            aggressive_cleanup()  # Cleanup on error
    
    return answers

# --- MAIN ENDPOINT ---
@app.post("/hackrx/run", response_model=HackRxResponse, dependencies=[Depends(verify_token)])
async def run_submission(request: HackRxRequest):
    """Render-optimized main processing endpoint"""
    start_time = time.time()
    
    try:
        logging.info(f"🚀 Processing {len(request.questions)} questions...")
        
        # Monitor initial memory
        initial_mem = monitor_memory()
        logging.info(f"📊 Initial memory: {initial_mem:.1f}MB")
        
        # Process document
        documents = render_optimized_document_processing(request.documents)
        
        # Create vector store
        vector_store = create_memory_efficient_vector_store(documents)
        
        # Create RAG chain
        rag_chain = create_render_optimized_chain(vector_store)
        
        # Process questions
        answers = await process_questions_efficiently(request.questions, rag_chain)
        
        # Final cleanup
        aggressive_cleanup()
        
        # Report performance
        total_time = time.time() - start_time
        final_mem = monitor_memory()
        logging.info(f"✅ Completed in {total_time:.1f}s, memory: {final_mem:.1f}MB")
        
        return HackRxResponse(answers=answers)
        
    except HTTPException:
        aggressive_cleanup()
        raise
    except Exception as e:
        aggressive_cleanup()
        logging.error(f"❌ Processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# --- UTILITY ENDPOINTS ---
@app.get("/health")
async def health_check():
    mem = monitor_memory()
    return {
        "status": "healthy",
        "version": "10.0.0",
        "memory_mb": round(mem, 1),
        "message": "Render-Optimized RAG System"
    }

@app.get("/")
async def root():
    return {
        "message": "Render-Optimized RAG System",
        "version": "10.0.0",
        "optimizations": [
            "Aggressive memory management",
            "Batch processing with cleanup",
            "Limited document chunks (100 max)",
            "Smart caching (2 entries max)",
            "Memory monitoring",
            "Render-specific tuning"
        ],
        "memory_limit": "512MB",
        "platform": "Render"
    }

@app.get("/memory-status")
async def memory_status():
    """Detailed memory status for monitoring"""
    mem = monitor_memory()
    return {
        "memory_mb": round(mem, 1),
        "memory_percent": round((mem / 512) * 100, 1),
        "status": "OK" if mem < 350 else "WARNING" if mem < 450 else "CRITICAL",
        "models_loaded": list(_models.keys()),
        "vector_stores_cached": len(_vector_store_cache),
        "cache_limit": _cache_limit,
        "recommendations": [
            "Clear cache if memory > 80%" if mem > 400 else "Memory usage normal",
            f"Current usage: {(mem/512)*100:.1f}% of 512MB limit"
        ]
    }

@app.post("/clear-cache")
async def clear_cache():
    """Emergency cache clearing endpoint"""
    cleared_models = len(_models)
    cleared_cache = len(_vector_store_cache)
    
    _models.clear()
    _vector_store_cache.clear()
    _namespace_cache.clear()
    aggressive_cleanup()
    
    new_mem = monitor_memory()
    
    return {
        "message": "Cache cleared successfully",
        "models_cleared": cleared_models,
        "cache_cleared": cleared_cache,
        "memory_after_cleanup_mb": round(new_mem, 1),
        "status": "success"
    }

# --- MAIN ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print("🚀 Starting Render-Optimized RAG System...")
    print("💾 Memory Limit: 512MB")
    print("🔧 Optimizations: Aggressive cleanup + Smart caching + Memory monitoring")
    print(f"🎯 Port: {port}")
    print("📚 Docs: /docs")
    print("💚 Health: /health")
    print("📊 Memory: /memory-status")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        workers=1,
        reload=False,
        access_log=False,  # Reduce logging overhead
        log_level="info"
    )