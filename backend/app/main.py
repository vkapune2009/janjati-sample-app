from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse 
import os
import uuid
from datetime import datetime
import logging
import sys

# Set up main logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("janjati_system.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import our custom modules
from app.services.database import init_db, save_session
from app.models.schemas import InitRequest, QuestionRequest
from app.services.vector_db import find_best_context, add_document_to_index
from app.services.llm_service import get_llm_answer

# Initialize Database
init_db()

app = FastAPI(title="Janjati Knowledge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Observability: Latency Monitoring Middleware ---
import time

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    # Log the route and how many milliseconds it took
    logger.info(f"API Request - {request.method} {request.url.path} completed in {process_time * 1000:.2f}ms")
    response.headers["X-Process-Time"] = str(process_time)
    return response

# --- Serve the HTML Frontend ---
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

# --- API 1: INIT ---
@app.post("/api/init")
def init_lms(request: InitRequest):
    uid = str(uuid.uuid4())
    logger.info(f"Init Request - Username: {request.username}, Method: {request.method}")

    # Save to SQLite using database.py
    save_session(uid, request.username, request.reqdatetime)

    return {
        "method": "init_lms_res",
        "username": request.username,
        "uniqueid": uid,
        "reqdatetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# --- API 2: QUESTION ---
@app.post("/api/ask")
def ask_question(request: QuestionRequest):
    logger.info(f"Question Request - Username: {request.username}, UID: {request.uniqueid}, Question: {request.question}")

    # Step 1: Use Vector DB to find context
    best_context = find_best_context(request.question)
    
    # Step 2: Use LLM Service to generate answer
    bot_reply = get_llm_answer(best_context, request.question)

    return {
        "method": "init_lms_answer",
        "username": request.username,
        "response": bot_reply,
        "reqdatetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# --- API 3: UPLOAD ---
@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...), admin_secret: str = Form(...)):
    logger.info(f"Upload Request - Filename: {file.filename}")
    
    # --- Security: Secret Verification ---
    # In production, use secrets.compare_digest for timing attack protection
    import secrets
    from dotenv import load_dotenv
    load_dotenv()
    
    expected_secret = os.getenv("ADMIN_SECRET", "supersecret123")
    
    if not secrets.compare_digest(admin_secret, expected_secret):
        logger.warning(f"Failed upload attempt. Invalid secret provided.")
        raise HTTPException(status_code=403, detail="Unauthorized. Invalid Admin Secret.")
    
    try:
        # Read the file bytes
        file_bytes = await file.read()
        
        # Process and store the document in vector DB
        num_chunks = add_document_to_index(file_bytes, file.filename)
        
        return {
            "message": f"Successfully processed {file.filename}",
            "chunks_added": num_chunks
        }
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))