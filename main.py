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
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone, ServerlessSpec

# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EXPECTED_BEARER_TOKEN = "612aeb3ebe9d63cfdb21e3f7d679fcebde54f7c1283c92b7937ea72c10c966af"
PINECONE_INDEX_NAME = "hackrx" 
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

if not all([PINECONE_API_KEY, GOOGLE_API_KEY]):
    raise ValueError("API keys for Pinecone and Google are not set in the environment variables.")

app = FastAPI(
    title="Universal Hybrid Search RAG System",
    description="Domain-agnostic RAG with adaptive hybrid search for any document type",
    version="3.0.0"
)

class HackRxRequest(BaseModel):
    documents: str = Field(..., description="URL to the PDF document")
    questions: List[str] = Field(..., description="List of questions to ask about the document")

class HackRxResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers corresponding to the questions")

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

# Cache for embeddings and LLM
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
            logging.info("Loading universal embeddings model...")
            _embeddings_cache = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'batch_size': 8}
            )
            logging.info("✅ Universal embeddings loaded successfully")
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
                max_tokens=400
            )
            logging.info("✅ Universal LLM initialized successfully")
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
    def extract_dynamic_terms(documents: List[Document], top_k: int = 20) -> Dict[str, float]:
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
        
        # Calculate importance scores
        important_terms = {}
        total_words = len(words)
        
        for word, freq in word_freq.most_common(top_k * 3):
            if word not in stop_words and len(word) > 3:
                # TF-IDF-like scoring: frequency * rarity
                tf = freq / total_words
                # Simple rarity: longer words and specific patterns get higher scores
                rarity_score = len(word) / 10  # Longer words are more specific
                
                # Boost for capitalized terms (likely proper nouns or important concepts)
                if any(word.capitalize() in doc.page_content for doc in documents[:3]):
                    rarity_score *= 1.5
                
                # Boost for terms that appear with numbers (likely metrics/values)
                if any(re.search(rf'\b{word}\b.*?\d+|\d+.*?\b{word}\b', doc.page_content.lower()) 
                       for doc in documents[:5]):
                    rarity_score *= 1.3
                
                importance_score = tf * rarity_score * 100  # Scale for easier handling
                important_terms[word] = min(importance_score, 10.0)  # Cap at 10.0
        
        # Return top k terms
        sorted_terms = dict(sorted(important_terms.items(), key=lambda x: x[1], reverse=True)[:top_k])
        logging.info(f"Dynamic terms extracted: {list(sorted_terms.keys())[:10]}...")
        return sorted_terms

class UniversalHybridVectorStore:
    """Domain-agnostic hybrid search with adaptive term weighting"""
    
    def __init__(self, documents: List[Document], embeddings):
        self.documents = documents
        self.embeddings = embeddings
        self.document_embeddings = None
        
        # Analyze document domain and extract important terms
        self.domain = DocumentAnalyzer.detect_document_domain(documents)
        self.dynamic_terms = DocumentAnalyzer.extract_dynamic_terms(documents)
        
        self._precompute_embeddings()
        logging.info(f"✅ Universal hybrid store created: {len(documents)} docs, domain: {self.domain}")
    
    def _precompute_embeddings(self):
        """Precompute embeddings for semantic search"""
        try:
            texts = [doc.page_content for doc in self.documents]
            self.document_embeddings = self.embeddings.embed_documents(texts)
            logging.info("✅ Document embeddings precomputed")
        except Exception as e:
            logging.warning(f"Failed to precompute embeddings: {e}")
            self.document_embeddings = None
    
    def _adaptive_keyword_search(self, query: str, k: int = 15) -> List[Tuple[Document, float]]:
        """Adaptive keyword search using dynamically extracted terms"""
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        scored_docs = []
        for doc in self.documents:
            content_lower = doc.page_content.lower()
            content_words = set(re.findall(r'\b\w+\b', content_lower))
            
            # Basic keyword overlap
            overlap_score = len(query_words.intersection(content_words))
            
            # Exact phrase matching (always important)
            phrase_score = 8.0 if query_lower in content_lower else 0
            
            # Dynamic term boosting based on document-specific important terms
            dynamic_boost = 0
            for term, weight in self.dynamic_terms.items():
                if term in query_lower and term in content_lower:
                    dynamic_boost += weight
            
            # Numerical pattern matching (universal across domains)
            query_numbers = re.findall(r'\b\d+\b', query_lower)
            content_numbers = re.findall(r'\b\d+\b', content_lower)
            number_score = len(set(query_numbers).intersection(set(content_numbers))) * 3.0
            
            # Question-specific term boosting (adaptive)
            question_boost = 0
            question_patterns = ['what', 'how', 'when', 'where', 'why', 'which', 'who']
            if any(pattern in query_lower for pattern in question_patterns):
                # Boost informational content patterns
                info_patterns = ['definition', 'explanation', 'description', 'meaning', 'purpose']
                for pattern in info_patterns:
                    if pattern in content_lower:
                        question_boost += 2.0
            
            total_score = overlap_score + phrase_score + dynamic_boost + number_score + question_boost
            scored_docs.append((doc, total_score))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:k]
    
    def _semantic_search(self, query: str, k: int = 15) -> List[Tuple[Document, float]]:
        """Universal semantic search (domain-agnostic)"""
        if self.document_embeddings is None:
            return []
        
        try:
            query_embedding = self.embeddings.embed_query(query)
            
            similarities = []
            for i, doc_embedding in enumerate(self.document_embeddings):
                dot_product = sum(a * b for a, b in zip(query_embedding, doc_embedding))
                magnitude_q = math.sqrt(sum(a * a for a in query_embedding))
                magnitude_d = math.sqrt(sum(a * a for a in doc_embedding))
                
                if magnitude_q > 0 and magnitude_d > 0:
                    similarity = dot_product / (magnitude_q * magnitude_d)
                else:
                    similarity = 0.0
                
                similarities.append((self.documents[i], similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:k]
            
        except Exception as e:
            logging.warning(f"Semantic search failed: {e}")
            return []
    
    def universal_hybrid_search(self, query: str, k: int = 8) -> List[Document]:
        """Universal hybrid search that adapts to any document domain"""
        
        # Get results from both search methods
        keyword_results = self._adaptive_keyword_search(query, k=12)
        semantic_results = self._semantic_search(query, k=12)
        
        logging.info(f"Adaptive keyword search: {len(keyword_results)} results")
        logging.info(f"Semantic search: {len(semantic_results)} results")
        
        # Adaptive weighting based on domain and query characteristics
        keyword_weight, semantic_weight = self._determine_search_weights(query)
        
        # Combine results with adaptive weighting
        combined_scores = {}
        
        for doc, score in keyword_results:
            doc_id = id(doc)
            combined_scores[doc_id] = {
                'doc': doc,
                'keyword_score': score * keyword_weight,
                'semantic_score': 0.0
            }
        
        for doc, score in semantic_results:
            doc_id = id(doc)
            if doc_id in combined_scores:
                combined_scores[doc_id]['semantic_score'] = score * semantic_weight
            else:
                combined_scores[doc_id] = {
                    'doc': doc,
                    'keyword_score': 0.0,
                    'semantic_score': score * semantic_weight
                }
        
        # Calculate final scores and sort
        final_results = []
        for doc_data in combined_scores.values():
            total_score = doc_data['keyword_score'] + doc_data['semantic_score']
            final_results.append((doc_data['doc'], total_score))
        
        final_results.sort(key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, score in final_results[:k]]
        
        if final_results:
            logging.info(f"Universal hybrid search: Top score {final_results[0][1]:.3f}")
        
        return top_docs
    
    def _determine_search_weights(self, query: str) -> Tuple[float, float]:
        """Adaptively determine keyword vs semantic weights based on query characteristics"""
        query_lower = query.lower()
        
        # Default balanced weights
        keyword_weight = 0.5
        semantic_weight = 0.5
        
        # Adjust based on query patterns
        if any(term in query_lower for term in ['what is', 'define', 'meaning', 'explanation']):
            # Conceptual questions benefit more from semantic search
            semantic_weight = 0.7
            keyword_weight = 0.3
        elif any(term in query_lower for term in ['how much', 'how many', 'percentage', 'rate']):
            # Numerical questions benefit from keyword precision
            keyword_weight = 0.7
            semantic_weight = 0.3
        elif re.search(r'\b\d+\b', query_lower):
            # Queries with numbers need keyword precision
            keyword_weight = 0.6
            semantic_weight = 0.4
        elif len(query_lower.split()) > 10:
            # Complex queries benefit from semantic understanding
            semantic_weight = 0.6
            keyword_weight = 0.4
        
        return keyword_weight, semantic_weight

def load_and_chunk_document_from_url(url: str) -> List[Document]:
    """Universal document loading with adaptive chunking"""
    logging.info(f"Loading document for universal processing: {url}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        temp_file_path = "temp_document.pdf"
        with open(temp_file_path, "wb") as temp_f:
            temp_f.write(response.content)
        
        # Universal chunking strategy (not domain-specific)
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load_and_split(
            text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=1000,  # Balanced for most document types
                chunk_overlap=200,  # Sufficient overlap for context
                separators=["\n\n", "\n", ". ", ".", " ", ""]
            )
        )
        
        logging.info(f"Universal chunking: {len(docs)} chunks created")
        
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        return docs

    except Exception as e:
        logging.error(f"Failed to load document: {e}")
        if os.path.exists("temp_document.pdf"):
            os.remove("temp_document.pdf")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {e}")

def clear_pinecone_index():
    """Clear Pinecone index for fresh processing"""
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        existing_indexes = pc.list_indexes().names()
        
        if PINECONE_INDEX_NAME in existing_indexes:
            index = pc.Index(PINECONE_INDEX_NAME)
            index.delete(delete_all=True)
            logging.info("✅ Vector index cleared")
            time.sleep(2)
    except Exception as e:
        logging.warning(f"Failed to clear index: {e}")

def create_universal_vector_store(documents: List[Document]) -> UniversalHybridVectorStore:
    """Create universal hybrid vector store"""
    logging.info("Creating universal hybrid vector store...")
    
    try:
        clear_pinecone_index()
        embeddings = get_cached_embeddings()
        
        # Create adaptive hybrid store
        hybrid_store = UniversalHybridVectorStore(documents, embeddings)
        
        logging.info("✅ Universal hybrid vector store ready")
        return hybrid_store
        
    except Exception as e:
        logging.error(f"Failed to create universal vector store: {e}")
        raise HTTPException(status_code=500, detail=f"Vector store creation failed: {e}")

def create_universal_rag_chain(vector_store: UniversalHybridVectorStore):
    """Create universal RAG chain with domain-agnostic prompting"""
    logging.info("Creating universal RAG chain...")
    
    try:
        llm = get_cached_llm()
        
        # Universal prompt template (no domain assumptions)
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
        
        # Universal retrieval function
        def universal_retriever(query: str) -> List[Document]:
            docs = vector_store.universal_hybrid_search(query, k=10)
            logging.info(f"Universal retrieval: {len(docs)} documents for query")
            return docs
        
        # Build universal RAG chain
        rag_chain = (
            {"context": lambda x: format_docs(universal_retriever(x)), "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        
        logging.info("✅ Universal RAG chain created")
        return rag_chain
        
    except Exception as e:
        logging.error(f"Failed to create universal RAG chain: {e}")
        raise HTTPException(status_code=500, detail=f"RAG chain creation failed: {e}")

def format_docs(docs: List[Document]) -> str:
    """Universal document formatting"""
    formatted_sections = []
    for i, doc in enumerate(docs, 1):
        formatted_sections.append(f"--- Section {i} ---\n{doc.page_content}")
    return "\n\n".join(formatted_sections)

@app.post("/hackrx/run", response_model=HackRxResponse, dependencies=[Depends(verify_token)])
async def run_submission(request: HackRxRequest):
    logging.info("🚀 Processing with universal hybrid search...")
    start_time = time.time()
    
    try:
        # Universal document processing
        documents = load_and_chunk_document_from_url(request.documents)
        vector_store = create_universal_vector_store(documents)
        rag_chain = create_universal_rag_chain(vector_store)
        
        # Process questions universally
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
        logging.info(f"✅ Universal processing complete in {total_time:.2f}s")
        
        gc.collect()
        return HackRxResponse(answers=answers)
        
    except Exception as e:
        logging.error(f"Universal processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/health")
async def health_check():
    memory_usage = get_memory_usage()
    return {
        "status": "healthy", 
        "message": "Universal Hybrid Search RAG API",
        "version": "3.0.0",
        "memory_mb": f"{memory_usage:.1f}" if memory_usage > 0 else "monitoring_unavailable",
        "features": [
            "Domain-agnostic hybrid search",
            "Dynamic term extraction",
            "Adaptive search weighting",
            "Universal document processing"
        ],
        "embeddings_loaded": _embeddings_cache is not None,
        "llm_loaded": _llm_cache is not None
    }

@app.get("/")
async def root():
    return {
        "message": "Universal Hybrid Search RAG System",
        "version": "3.0.0",
        "features": [
            "Automatic domain detection",
            "Dynamic important term extraction", 
            "Adaptive keyword + semantic search",
            "Universal document compatibility",
            "No hardcoded domain assumptions"
        ],
        "docs": "/docs",
        "health": "/health"
    }

@app.on_event("startup")
async def startup_event():
    logging.info("🚀 Starting Universal RAG System...")
    try:
        get_cached_embeddings()
        logging.info("✅ Universal embeddings ready")
    except Exception as e:
        logging.warning(f"⚠️ Embeddings will load on demand: {e}")
    
    memory_usage = get_memory_usage()
    logging.info(f"📊 Startup memory: {memory_usage:.1f}MB")
    logging.info("🌍 Universal system ready for any document domain")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print("🚀 Universal Hybrid Search RAG System")
    print("🌍 Domain-agnostic | Works with any document type")
    print("🔍 Adaptive hybrid search with dynamic term extraction")
    print(f"🎯 Running on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
