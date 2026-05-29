import os
import shutil
import tempfile
import uvicorn
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from translator import init_translator, translate_nt_text
from asr_service import ASRService

# Lifespan event to load translation dictionary RAG on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Initializing dictionary RAG database...")
    try:
        init_translator()
        print("[Startup] RAG database initialized successfully.")
    except Exception as e:
        print(f"[Startup] Warning: RAG database initialization failed: {e}")
        print("[Startup] App will continue, but translation might fail until configured.")
    
    # Initialize default ASR Service
    global asr_service
    default_model = os.getenv("ASR_MODEL_PATH", "build/onnx-web/merged/whisper-lora-nghe-tinh")
    asr_service = ASRService(default_model)
    
    # Warm up / pre-load Whisper model during startup so first web call is instant
    print("[Startup] Pre-loading Whisper ASR model...")
    try:
        asr_service.load_model()
        print("[Startup] Whisper ASR model pre-loaded successfully.")
    except Exception as e:
        print(f"[Startup] Warning: Whisper ASR model pre-loading failed: {e}")
        print("[Startup] ASR model will load on first request.")
    
    yield
    print("[Shutdown] Cleaning up...")

app = FastAPI(
    title="Nghe An & Ha Tinh ASR & Translation",
    description="ASR Voice-to-Text and LLM dialect translator",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for local development ease
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global ASR service instance
asr_service = None

# API models
class TranslateRequest(BaseModel):
    text: str

class TranslateResponse(BaseModel):
    original_text: str
    translated_text: str
    matched_terms: list

@app.post("/api/translate", response_model=TranslateResponse)
async def translate(request: TranslateRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    print(f"[API] Translate request: {request.text}")
    translation, matched_terms = translate_nt_text(request.text)
    return TranslateResponse(
        original_text=request.text,
        translated_text=translation,
        matched_terms=matched_terms
    )

@app.post("/api/asr")
async def transcribe_audio(file: UploadFile = File(...), model_name: str = Form(None)):
    global asr_service
    
    # If a specific model is selected, re-initialize or update path
    if model_name:
        target_path = Path("build/onnx-web/merged") / model_name
        if target_path.exists():
            if asr_service.model_path != str(target_path.resolve()):
                print(f"[API] Switching ASR model to {model_name}...")
                asr_service = ASRService(str(target_path))
        else:
            print(f"[API] Requested model {model_name} not found at {target_path}, using active model.")

    # Save uploaded file to a temporary file
    temp_dir = Path("build/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = Path(file.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=temp_dir) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = Path(temp_file.name)

    try:
        # Transcribe audio using ASR Service
        transcript = asr_service.transcribe(str(temp_path))
        return {"transcript": transcript, "model_used": Path(asr_service.model_path).name}
    except Exception as e:
        print(f"[API] ASR Error: {e}")
        raise HTTPException(status_code=500, detail=f"ASR transcription failed: {str(e)}")
    finally:
        # Ensure cleanup of temp file
        if temp_path.exists():
            temp_path.unlink()

@app.get("/api/models")
async def get_models():
    # Scan merged models directory
    merged_dir = Path("build/onnx-web/merged")
    models = []
    if merged_dir.exists():
        for item in merged_dir.iterdir():
            if item.is_dir() and (item / "config.json").exists():
                models.append(item.name)
    
    # Fallback if directory does not exist or empty
    if not models:
        models = ["whisper-lora-nghe-tinh", "whisper-dora-nghe-tinh"]
        
    current_model = Path(asr_service.model_path).name if asr_service else "whisper-lora-nghe-tinh"
    return {"models": models, "active": current_model}

# Serve static frontend files
static_dir = Path("static")
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    print(f"[Server] Starting server at http://localhost:{port}...")
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=True)
