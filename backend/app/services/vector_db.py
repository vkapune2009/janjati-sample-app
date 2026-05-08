import logging
import os
import io
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "vector_store")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Check if we are running on Render (with a HF_TOKEN) to save memory
hf_token = os.getenv("HF_TOKEN")

if hf_token:
    logger.info("HF_TOKEN found. Using HuggingFace API for embeddings to save RAM...")
    from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=hf_token, model_name=MODEL_NAME)
else:
    logger.info("Loading local HuggingFace embeddings model (requires lots of RAM)...")
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
logger.info("Embeddings model loaded successfully.")

vector_store = None

fallback_documents = [
    "Python is a programming language used for AI and web development.",
    "Machine Learning is a subset of AI that allows systems to learn from data.",
    "JavaScript is mainly used for building interactive web applications.",
    "Birsa Munda was an Indian tribal independence activist and folk hero."
]

def load_vector_store():
    global vector_store
    # Check if the LangChain FAISS index exists
    if os.path.exists(SAVE_DIR) and os.path.exists(os.path.join(SAVE_DIR, "index.faiss")):
        try:
            # allow_dangerous_deserialization is required for FAISS to load pickle files in newer LangChain versions
            vector_store = FAISS.load_local(SAVE_DIR, embeddings, allow_dangerous_deserialization=True)
            logger.info("Vector store loaded successfully.")
        except Exception as exc:
            logger.error(f"Unable to load vector store: {exc}")
    else:
        logger.warning("Vector store not found. A new one will be created upon first upload.")

load_vector_store()

def find_best_context(query: str, top_k: int = 3):
    logger.info("Vector DB: retrieving best context for query...")
    
    if vector_store is None:
        logger.info("Using fallback documents for best context.")
        return fallback_documents[0]
        
    try:
        # LangChain makes similarity search extremely easy
        results = vector_store.similarity_search(query, k=top_k)
        if not results:
            return fallback_documents[0]
            
        formatted_results = []
        for doc in results:
            name = doc.metadata.get("name", "Unknown")
            section = doc.metadata.get("section", "General")
            formatted_results.append(f"{section} ({name}): {doc.page_content}")
            
        return "\n\n".join(formatted_results)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return fallback_documents[0]

def add_document_to_index(file_bytes: bytes, filename: str):
    logger.info(f"Processing uploaded document: {filename}")
    global vector_store
    
    # 1. Extract text
    text = ""
    if filename.lower().endswith(".pdf"):
        pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    else:
        text = file_bytes.decode("utf-8", errors="ignore")
        
    if not text.strip():
        raise ValueError("Could not extract any text from the document.")

    # 2. Chunk text using LangChain
    logger.info("Chunking text...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    new_chunks = splitter.split_text(text)
    
    if not new_chunks:
        raise ValueError("Document yielded 0 chunks.")

    # 3. Create LangChain Document objects
    documents = [
        Document(page_content=chunk, metadata={"name": filename, "section": "Uploaded Document"})
        for chunk in new_chunks
    ]

    # 4. Add to LangChain FAISS
    logger.info(f"Adding {len(documents)} chunks to vector store...")
    if vector_store is None:
        # Initialize new store if it doesn't exist
        vector_store = FAISS.from_documents(documents, embeddings)
    else:
        # Add to existing store
        vector_store.add_documents(documents)
        
    # Save back to disk securely
    os.makedirs(SAVE_DIR, exist_ok=True)
    vector_store.save_local(SAVE_DIR)
        
    logger.info(f"Successfully added {len(documents)} chunks to vector store.")
    return len(documents)
