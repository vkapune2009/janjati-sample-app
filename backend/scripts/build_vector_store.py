import os
import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BACKEND_DIR, "knowledge_data")
SAVE_DIR = os.path.join(BACKEND_DIR, "vector_store")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_knowledge_data():
    documents = []
    if not os.path.exists(DATA_DIR):
        print(f"Error: knowledge directory not found: {DATA_DIR}")
        return documents

    for hero_folder in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, hero_folder)
        if not os.path.isdir(folder_path):
            continue

        json_path = os.path.join(folder_path, "metadata.json")
        if not os.path.exists(json_path):
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            hero_entries = json.load(f)

        for entry in hero_entries:
            text = entry.get("text", "").strip()
            if not text:
                continue

            metadata = entry.get("metadata", {}).copy()
            metadata["section"] = entry.get("section", "General")
            
            # Convert to official LangChain Document
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)

    return documents

def build_vector_store():
    documents = load_knowledge_data()
    if not documents:
        print("No knowledge data found. Add JSON files under backend/knowledge_data.")
        return

    print(f"Found {len(documents)} knowledge chunks from JSON files.")
    print("Loading HuggingFace Embeddings model...")
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
        embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=hf_token, model_name=MODEL_NAME)
    else:
        embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    
    # Check if a LangChain vector store already exists from UI uploads
    if os.path.exists(SAVE_DIR) and os.path.exists(os.path.join(SAVE_DIR, "index.faiss")):
        print("Existing vector store found! Appending old data to it...")
        vector_store = FAISS.load_local(SAVE_DIR, embeddings, allow_dangerous_deserialization=True)
        vector_store.add_documents(documents)
    else:
        print("No existing vector store found. Creating a brand new one...")
        os.makedirs(SAVE_DIR, exist_ok=True)
        vector_store = FAISS.from_documents(documents, embeddings)

    # Save to disk
    vector_store.save_local(SAVE_DIR)
    print(f"✅ Success! Vector store updated and saved in {SAVE_DIR}.")

if __name__ == "__main__":
    build_vector_store()
