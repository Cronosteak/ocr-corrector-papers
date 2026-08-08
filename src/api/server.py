"""
server.py — FastAPI REST service for correcting OCR text.
"""

import os

from dotenv import load_dotenv
from fastapi import FastAPI

from src.api.schemas import CorrectionRequest, CorrectionResponse
from src.model.predict import correct_text, load_model

load_dotenv()

app = FastAPI(
    title="OCR Corrector API",
    description="Service that corrects OCR text from academic papers",
    version="0.1.0",
)

# The model is loaded once at server startup
MODEL_PATH = os.getenv("MODEL_PATH", "models/ocr-corrector")
model = None
tokenizer = None


@app.on_event("startup")
async def startup_event():
    """Load the model on startup."""
    global model, tokenizer
    try:
        model, tokenizer = load_model(MODEL_PATH)
    except Exception as e:
        print(f"Warning: could not load the model: {e}")


@app.post("/correct", response_model=CorrectionResponse)
async def correct_ocr_text(request: CorrectionRequest) -> CorrectionResponse:
    """
    Correct a noisy OCR text.

    - **text**: OCR text to correct
    """
    corrected = correct_text(
        text=request.text,
        model=model,
        tokenizer=tokenizer,
        model_path=MODEL_PATH,
    )
    return CorrectionResponse(
        original=request.text,
        corrected=corrected,
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": model is not None}


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
