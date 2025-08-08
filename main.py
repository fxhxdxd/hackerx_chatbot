import os
import io
import logging
import time
import gc
from typing import List, Tuple, Optional, Union, Dict, Any
import re
import math
from collections import Counter
import hashlib
import uuid

import requests
import uvicorn
from dotenv import load_dotenv

from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from pinecone import Pinecone, ServerlessSpec

# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EXPECTED_BEARER_TOKEN = "612aeb3ebe9d63cfdb21e3f7d679fcebde54f7c1283c92b7937ea72c10c966af"
PINECONE_INDEX_NAME = "universal-rag"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Memory optimization config
class MemoryConfig:
    store_full_text_in_metadata: bool = True
    local_snippet_size: int = 2000
    max_terms_per_chunk: int = 10
    embedding_batch_size: int = 8
    pinecone_batch_size: int = 16
    enable_aggressive_gc: bool = True
    namespace_ttl_hours: int = 24

CONFIG = MemoryConfig()

if not all([PINECONE_API_KEY, GOOGLE_API_KEY]):
    raise ValueError("API keys for Pinecone and Google are not set in environment variables")

app = FastAPI(
    title="Memory-Optimized Universal Hybrid RAG System",
    description="Domain-agnostic RAG with streaming embeddings, hybrid search, and minimal memory footprint",
    version="4.0.0"
)

class HackRxRequest(BaseModel):
    documents: str = Field(..., description="URL to the PDF document")
    questions: List[str] = Field(..., description="List of questions to ask about the document")

class HackRxResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers corresponding to the questions")

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

# Global model cache
_embeddings_cache = None
_llm_cache = None

def get_memory_usage() -> float:
    """Get approximate memory usage without psutil"""
    try:
        gc.collect()
        try:
            with open('/proc/self/status') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        memory_kb = int(line.split()[1])
                        return memory_kb / 1024
        except:
            pass
        return 0.0
    except:
        return 0.0

def get_cached_embeddings():
    global _embeddings_cache
    if _embeddings_cache is None:
        try:
            logging.info("Loading embeddings model for streaming...")
            _embeddings_cache = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'batch_size': CONFIG.embedding_batch_size}
            )
            logging.info("✅ Embeddings model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load embeddings: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load embeddings: {e}")
    return _embeddings_cache

def get_cached_llm():
    global _llm_cache
    if _llm_cache is None:
        try:
            _llm_cache = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                temperature=0.1,
                google_api_key=GOOGLE_API_KEY,
                max_tokens=500
            )
            logging.info("✅ LLM initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize LLM: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize LLM: {e}")
    return _llm_cache

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

class DocumentAnalyzer:
    """Adaptive document analysis for domain detection and term importance"""
    
    @staticmethod
    def detect_document_domain(documents: List[Document]) -> str:
        """Automatically detect document domain/type"""
        all_text = " ".join([doc.page_content for doc in documents[:5]]).lower()
        
        domain_indicators = {
            'insurance': ['policy', 'premium', 'coverage', 'claim', 'deductible', 'beneficiary'],
            'legal': ['contract', 'agreement', 'clause', 'whereas', 'party', 'terms'],
            'medical': ['patient', 'diagnosis', 'treatment', 'medical', 'clinical', 'therapy'],
            'financial': ['investment', 'portfolio', 'asset', 'liability', 'revenue', 'profit'],
            'technical': ['system', 'implementation', 'configuration', 'specification', 'architecture'],
            'academic': ['research', 'study', 'analysis', 'methodology', 'conclusion', 'abstract'],
            'regulatory': ['regulation', 'compliance', 'requirement', 'standard', 'guideline', 'procedure']
        }
        
        domain_scores = {}
        for domain, indicators in domain_indicators.items():
            score = sum(1 for indicator in indicators if indicator in all_text)
            domain_scores[domain] = score
        
        detected_domain = max(domain_scores, key=domain_scores.get) if domain_scores else 'general'
        confidence = domain_scores[detected_domain] / len(domain_indicators[detected_domain])
        
        logging.info(f"Detected domain: {detected_domain} (confidence: {confidence:.2f})")
        return detected_domain
    
    @staticmethod
    def extract_dynamic_terms(documents: List[Document], top_k: int = CONFIG.max_terms_per_chunk) -> Dict[str, float]:
        """Extract important terms dynamically from the document corpus"""
        all_text = " ".join([doc.page_content for doc in documents]).lower()
        
        # Extract all words and calculate frequency
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text)
        word_freq = Counter(words)
        
        # Filter out common stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was',
            'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now',
            'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she',
            'too', 'use', 'will', 'with', 'have', 'this', 'that', 'they', 'from', 'been', 'have',
            'their', 'said', 'each', 'which', 'there', 'what', 'were', 'when', 'your', 'than'
        }
        
        important_terms = {}
        total_words = len(words)
        
        for word, freq in word_freq.most_common(top_k * 3):
            if word not in stop_words and len(word) > 3:
                tf = freq / total_words
                rarity_score = len(word) / 10
                
                # Boost for capitalized terms
                if any(word.capitalize() in doc.page_content for doc in documents[:3]):
                    rarity_score *= 1.5
                
                # Boost for terms that appear with numbers
                if any(re.search(rf'\b{word}\b.*?\d+|\d+.*?\b{word}\b', doc.page_content.lower()) 
                       for doc in documents[:5]):
                    rarity_score *= 1.3
                
                importance_score = tf * rarity_score * 100
                important_terms[word] = min(importance_score, 10.0)
        
        sorted_terms = dict(sorted(important_terms.items(), key=lambda x: x[1], reverse=True)[:top_k])
        logging.info(f"Dynamic terms extracted: {list(sorted_terms.keys())[:10]}...")
        return sorted_terms

class ChunkMetadata:
    """Lightweight metadata for each document chunk"""
    def __init__(self, chunk_id: str, snippet: str, doc_id: str, source: str, important_terms: List[str]):
        self.chunk_id = chunk_id
        self.snippet = snippet[:CONFIG.local_snippet_size]  # Truncate to limit
        self.doc_id = doc_id
        self.source = source
        self.important_terms = important_terms

class LightweightInvertedIndex:
    """Memory-efficient inverted index for keyword search"""
    
    def __init__(self):
        self.term_to_chunks: Dict[str, List[str]] = {}
        self.chunk_metadata: Dict[str, ChunkMetadata] = {}
    
    def add_chunk(self, chunk_id: str, metadata: ChunkMetadata):
        """Add chunk to inverted index"""
        self.chunk_metadata[chunk_id] = metadata
        
        # Add important terms to inverted index
        for term in metadata.important_terms:
            if term not in self.term_to_chunks:
                self.term_to_chunks[term] = []
            self.term_to_chunks[term].append(chunk_id)
    
    def search_keywords(self, query: str) -> List[str]:
        """Fast keyword search returning chunk IDs"""
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        candidate_chunks = set()
        
        for word in query_words:
            if word in self.term_to_chunks:
                candidate_chunks.update(self.term_to_chunks[word])
        
        return list(candidate_chunks)
    
    def get_memory_footprint(self) -> int:
        """Estimate memory usage in bytes"""
        import sys
        size = sys.getsizeof(self.term_to_chunks)
        size += sys.getsizeof(self.chunk_metadata)
        return size

class StreamingPineconeManager:
    """Manages streaming embeddings to Pinecone with memory optimization"""
    
    def __init__(self, index_name: str):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = index_name
        self.namespace = f"doc_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        self.embeddings = get_cached_embeddings()
        
        # Ensure index exists
        self._ensure_index_exists()
        self.index = self.pc.Index(index_name)
        
        logging.info(f"Using namespace: {self.namespace}")
    
    def _ensure_index_exists(self):
        """Ensure Pinecone index exists"""
        try:
            existing_indexes = self.pc.list_indexes().names()
            if self.index_name not in existing_indexes:
                logging.info(f"Creating Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=384,  # all-MiniLM-L6-v2 dimension
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                # Wait for index to be ready
                time.sleep(10)
        except Exception as e:
            logging.warning(f"Index creation/check failed: {e}")
    
    def stream_embeddings(self, documents: List[Document]) -> LightweightInvertedIndex:
        """Stream embeddings to Pinecone in batches"""
        inverted_index = LightweightInvertedIndex()
        domain = DocumentAnalyzer.detect_document_domain(documents)
        dynamic_terms = DocumentAnalyzer.extract_dynamic_terms(documents)
        
        total_chunks = len(documents)
        processed = 0
        
        # Process in batches
        for i in range(0, total_chunks, CONFIG.embedding_batch_size):
            batch_docs = documents[i:i + CONFIG.embedding_batch_size]
            
            # Generate embeddings for batch
            batch_texts = [doc.page_content for doc in batch_docs]
            batch_embeddings = self.embeddings.embed_documents(batch_texts)
            
            # Prepare vectors for Pinecone
            vectors_to_upsert = []
            for j, (doc, embedding) in enumerate(zip(batch_docs, batch_embeddings)):
                chunk_id = f"chunk_{i+j}"
                
                # Extract important terms for this chunk
                chunk_terms = self._extract_chunk_terms(doc.page_content, dynamic_terms)
                
                # Create lightweight metadata
                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    snippet=doc.page_content[:CONFIG.local_snippet_size],
                    doc_id=f"doc_{domain}",
                    source="uploaded_document",
                    important_terms=chunk_terms
                )
                
                # Add to inverted index
                inverted_index.add_chunk(chunk_id, metadata)
                
                # Prepare Pinecone vector
                pinecone_metadata = {
                    "text": doc.page_content[:40000] if CONFIG.store_full_text_in_metadata else "",  # Pinecone limit
                    "chunk_id": chunk_id,
                    "domain": domain
                }
                
                vectors_to_upsert.append({
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": pinecone_metadata
                })
            
            # Upsert to Pinecone
            try:
                self.index.upsert(vectors=vectors_to_upsert, namespace=self.namespace)
                processed += len(batch_docs)
                logging.info(f"Streamed batch {i//CONFIG.embedding_batch_size + 1}: {processed}/{total_chunks} chunks")
            except Exception as e:
                logging.error(f"Failed to upsert batch {i}: {e}")
            
            # Immediate memory cleanup
            del batch_embeddings, vectors_to_upsert
            if CONFIG.enable_aggressive_gc:
                gc.collect()
        
        logging.info(f"✅ Streamed {total_chunks} chunks to Pinecone. Inverted index: {inverted_index.get_memory_footprint()} bytes")
        return inverted_index
    
    def _extract_chunk_terms(self, text: str, dynamic_terms: Dict[str, float]) -> List[str]:
        """Extract important terms from a chunk"""
        text_lower = text.lower()
        chunk_terms = []
        
        for term, weight in dynamic_terms.items():
            if term in text_lower:
                chunk_terms.append(term)
        
        return chunk_terms[:CONFIG.max_terms_per_chunk]
    
    def semantic_search(self, query: str, top_k: int = 10) -> List[Document]:
        """Perform semantic search via Pinecone"""
        try:
            query_embedding = self.embeddings.embed_query(query)
            
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=self.namespace,
                include_metadata=True
            )
            
            documents = []
            for match in results.matches:
                content = match.metadata.get("text", "")
                if content:
                    doc = Document(
                        page_content=content,
                        metadata={"chunk_id": match.metadata.get("chunk_id"), "score": match.score}
                    )
                    documents.append(doc)
            
            logging.info(f"Semantic search returned {len(documents)} documents")
            return documents
            
        except Exception as e:
            logging.error(f"Semantic search failed: {e}")
            return []
    
    def cleanup_namespace(self):
        """Clean up the namespace"""
        try:
            self.index.delete(delete_all=True, namespace=self.namespace)
            logging.info(f"Cleaned up namespace: {self.namespace}")
        except Exception as e:
            logging.warning(f"Failed to cleanup namespace: {e}")

class MemoryOptimizedHybridRAG:
    """Memory-optimized hybrid RAG with Pinecone backend"""
    
    def __init__(self, documents: List[Document]):
        self.pinecone_manager = StreamingPineconeManager(PINECONE_INDEX_NAME)
        self.inverted_index = self.pinecone_manager.stream_embeddings(documents)
        
        logging.info("✅ Memory-optimized hybrid RAG initialized")
    
    def hybrid_search(self, query: str, k: int = 10) -> List[Document]:
        """Perform hybrid search combining keyword and semantic results"""
        
        # 1. Fast keyword search using inverted index
        keyword_chunk_ids = self.inverted_index.search_keywords(query)
        logging.info(f"Keyword search found {len(keyword_chunk_ids)} candidates")
        
        # 2. Semantic search via Pinecone
        semantic_docs = self.pinecone_manager.semantic_search(query, top_k=15)
        
        # 3. Combine results with adaptive weighting
        keyword_weight, semantic_weight = self._determine_search_weights(query)
        
        # Score and combine results
        combined_results = {}
        
        # Add keyword results
        for chunk_id in keyword_chunk_ids:
            if chunk_id in self.inverted_index.chunk_metadata:
                metadata = self.inverted_index.chunk_metadata[chunk_id]
                doc = Document(
                    page_content=metadata.snippet,
                    metadata={"chunk_id": chunk_id, "source": "keyword"}
                )
                combined_results[chunk_id] = {
                    "doc": doc,
                    "keyword_score": keyword_weight,
                    "semantic_score": 0.0
                }
        
        # Add semantic results
        for doc in semantic_docs:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id:
                if chunk_id in combined_results:
                    combined_results[chunk_id]["semantic_score"] = semantic_weight * doc.metadata.get("score", 0)
                else:
                    combined_results[chunk_id] = {
                        "doc": doc,
                        "keyword_score": 0.0,
                        "semantic_score": semantic_weight * doc.metadata.get("score", 0)
                    }
        
        # Sort by combined score
        scored_results = []
        for data in combined_results.values():
            total_score = data["keyword_score"] + data["semantic_score"]
            scored_results.append((data["doc"], total_score))
        
        scored_results.sort(key=lambda x: x[1], reverse=True)
        final_docs = [doc for doc, score in scored_results[:k]]
        
        logging.info(f"Hybrid search returned {len(final_docs)} documents")
        return final_docs
    
    def _determine_search_weights(self, query: str) -> Tuple[float, float]:
        """Adaptively determine keyword vs semantic weights"""
        query_lower = query.lower()
        
        keyword_weight = 0.5
        semantic_weight = 0.5
        
        if any(term in query_lower for term in ['what is', 'define', 'meaning']):
            semantic_weight = 0.7
            keyword_weight = 0.3
        elif any(term in query_lower for term in ['how much', 'how many', 'percentage']):
            keyword_weight = 0.7
            semantic_weight = 0.3
        elif re.search(r'\b\d+\b', query_lower):
            keyword_weight = 0.6
            semantic_weight = 0.4
        elif len(query_lower.split()) > 10:
            semantic_weight = 0.6
            keyword_weight = 0.4
        
        return keyword_weight, semantic_weight
    
    def cleanup(self):
        """Clean up resources"""
        self.pinecone_manager.cleanup_namespace()

def load_and_chunk_document_from_url(url: str) -> List[Document]:
    """Load and chunk document with memory optimization"""
    logging.info(f"Loading document: {url}")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        temp_file_path = "temp_document.pdf"
        with open(temp_file_path, "wb") as temp_f:
            temp_f.write(response.content)
        
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load_and_split(
            text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", ".", " ", ""]
            )
        )
        
        logging.info(f"Loaded {len(docs)} document chunks")
        
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        return docs

    except Exception as e:
        logging.error(f"Failed to load document: {e}")
        if os.path.exists("temp_document.pdf"):
            os.remove("temp_document.pdf")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {e}")

def create_universal_rag_chain(hybrid_rag: MemoryOptimizedHybridRAG):
    """Create universal RAG chain"""
    logging.info("Creating memory-optimized RAG chain...")
    
    try:
        llm = get_cached_llm()
        
        template = """You are an expert document analyst. Use the provided document sections to answer the question accurately and comprehensively.

INSTRUCTIONS:
- Base your answer strictly on the provided document sections
- Extract specific details, numbers, conditions, and requirements when mentioned
- Maintain accuracy and cite exact information from the text
- If the information is not available in the provided sections, clearly state this
- Provide clear, well-structured answers that directly address the question

DOCUMENT SECTIONS:
{context}

QUESTION: {question}

COMPREHENSIVE ANSWER:"""
        
        rag_prompt = PromptTemplate.from_template(template)
        
        def retriever(query: str) -> List[Document]:
            return hybrid_rag.hybrid_search(query, k=10)
        
        def format_docs(docs: List[Document]) -> str:
            formatted_sections = []
            for i, doc in enumerate(docs, 1):
                formatted_sections.append(f"--- Section {i} ---\n{doc.page_content}")
            return "\n\n".join(formatted_sections)
        
        rag_chain = (
            {"context": lambda x: format_docs(retriever(x)), "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        
        logging.info("✅ Memory-optimized RAG chain created")
        return rag_chain
        
    except Exception as e:
        logging.error(f"Failed to create RAG chain: {e}")
        raise HTTPException(status_code=500, detail=f"RAG chain creation failed: {e}")

@app.post("/hackrx/run", response_model=HackRxResponse, dependencies=[Depends(verify_token)])
async def run_submission(request: HackRxRequest):
    logging.info("🚀 Processing with memory-optimized hybrid RAG...")
    start_time = time.time()
    hybrid_rag = None
    
    try:
        # Load and process documents
        documents = load_and_chunk_document_from_url(request.documents)
        
        # Create memory-optimized hybrid RAG
        hybrid_rag = MemoryOptimizedHybridRAG(documents)
        rag_chain = create_universal_rag_chain(hybrid_rag)
        
        # Process questions
        answers = []
        for i, question in enumerate(request.questions, 1):
            logging.info(f"Processing question {i}: '{question[:60]}...'")
            
            try:
                answer = rag_chain.invoke(question)
                answers.append(answer.strip())
                logging.info(f"✅ Question {i} completed")
                
            except Exception as e:
                logging.error(f"Error on question {i}: {e}")
                answers.append(f"Error processing question: {str(e)}")
        
        total_time = time.time() - start_time
        memory_usage = get_memory_usage()
        logging.info(f"✅ Memory-optimized processing complete in {total_time:.2f}s. Memory: {memory_usage:.1f}MB")
        
        return HackRxResponse(answers=answers)
        
    except Exception as e:
        logging.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Cleanup resources
        if hybrid_rag:
            hybrid_rag.cleanup()
        if CONFIG.enable_aggressive_gc:
            gc.collect()

@app.get("/health")
async def health_check():
    memory_usage = get_memory_usage()
    return {
        "status": "healthy",
        "message": "Memory-Optimized Universal Hybrid RAG API",
        "version": "4.0.0",
        "memory_mb": f"{memory_usage:.1f}" if memory_usage > 0 else "monitoring_unavailable",
        "features": [
            "Streaming embeddings to Pinecone",
            "Lightweight inverted index",
            "Adaptive hybrid search",
            "Memory-optimized architecture",
            "Domain-agnostic processing"
        ],
        "embeddings_loaded": _embeddings_cache is not None,
        "llm_loaded": _llm_cache is not None
    }

@app.get("/")
async def root():
    return {
        "message": "Memory-Optimized Universal Hybrid RAG System",
        "version": "4.0.0",
        "architecture": [
            "Pinecone-backed semantic search",
            "Lightweight local inverted index",
            "Streaming batch processing",
            "Aggressive memory management",
            "Domain-agnostic design"
        ],
        "memory_optimizations": [
            "No local embedding matrix storage",
            "Streaming upsert to Pinecone",
            "Minimal local metadata",
            "Batch processing with cleanup",
            "Configurable memory limits"
        ],
        "docs": "/docs",
        "health": "/health"
    }

@app.on_event("startup")
async def startup_event():
    logging.info("🚀 Starting Memory-Optimized Universal RAG System...")
    try:
        get_cached_embeddings()
        logging.info("✅ Embeddings pre-loaded for streaming")
    except Exception as e:
        logging.warning(f"⚠️ Embeddings will load on demand: {e}")
    
    memory_usage = get_memory_usage()
    logging.info(f"📊 Startup memory: {memory_usage:.1f}MB")
    logging.info("🌍 Memory-optimized system ready for deployment")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print("🚀 Memory-Optimized Universal Hybrid RAG System")
    print("💾 Features: Streaming embeddings, lightweight index, minimal RAM")
    print("🔍 Hybrid search: Pinecone semantic + local keyword")
    print(f"🎯 Running on port {port}")
    print("⚡ Optimized for memory-constrained deployment")
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)  # Single worker for memory control