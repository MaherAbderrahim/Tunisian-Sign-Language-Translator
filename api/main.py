"""
FastAPI endpoints for TSL Translator.

Security / correctness fixes vs the original:
  - CORS: allow_origins is configurable via env (TSL_CORS_ORIGINS) and
    defaults to an explicit list; `allow_credentials` only set if origins are
    not "*".
  - Upload size limit: MAX_UPLOAD_BYTES enforced in /translate/frame.
  - Inference is run in a threadpool to avoid blocking the event loop.
  - History is already bounded in TSLTranslator (HISTORY_MAX_LEN).
"""

import os
from typing import Optional, Dict, List
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

import numpy as np
import cv2

from src.translation.translator import (
    TSLTranslator,
    translate_tsl_to_text,
    translate_text_to_tsl,
)
from src.languages.dictionary import TSLDictionary
from src.config.settings import MAX_UPLOAD_BYTES


# Create FastAPI app
app = FastAPI(
    title="TSL Translator API",
    description="Tunisian Sign Language (W7CH dataset) to Dialect Translation API",
    version="2.0.0",
)

# CORS — configurable via env, safe defaults
_default_origins = "http://localhost:8501,http://127.0.0.1:8501"
_cors_env = os.environ.get("TSL_CORS_ORIGINS", _default_origins)
_cors_origins = [
    o.strip() for o in _cors_env.split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=(_cors_origins != ["*"]),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global translator instance (initialized on startup)
translator: Optional[TSLTranslator] = None


class TranslationResponse(BaseModel):
    """Translation response model"""
    tsl: Optional[str] = None
    tunisian: Optional[str] = None
    french: Optional[str] = None
    english: Optional[str] = None
    arabic: Optional[str] = None
    arabic_msa: Optional[str] = None
    category: Optional[str] = None
    confidence: float
    rejected: bool = False


class TextTranslationRequest(BaseModel):
    """Text translation request"""
    text: str
    target_lang: str = "french"


class TextTranslationResponse(BaseModel):
    """Text translation response"""
    input: str
    tsl: Optional[str] = None
    tunisian: Optional[str] = None
    french: Optional[str] = None
    english: Optional[str] = None
    arabic: Optional[str] = None
    category: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize translator on startup."""
    global translator
    translator = TSLTranslator()
    print("TSL Translator initialized")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "TSL Translator API",
        "version": "2.0.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    ready = translator is not None
    return {"status": "healthy" if ready else "starting", "ready": ready}


@app.post("/translate/frame", response_model=TranslationResponse)
async def translate_frame(file: UploadFile = File(...)):
    """
    Translate a single frame (image).

    Accepts JPEG/PNG up to MAX_UPLOAD_BYTES. Inference runs in a threadpool
    so the event loop stays responsive.
    """
    if translator is None:
        raise HTTPException(status_code=503, detail="Translator not initialized")

    # Enforce upload size limit while reading
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024*1024)} MB)",
        )

    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image format")

    # Run CPU-bound inference in a thread (don't block the loop)
    result = await run_in_threadpool(translator.translate_frame, frame)

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return TranslationResponse(
        tsl=result.get("tsl"),
        tunisian=result.get("tunisian"),
        french=result.get("french"),
        english=result.get("english"),
        arabic=result.get("arabic"),
        category=result.get("category"),
        confidence=result.get("confidence", 0.0),
        rejected=result.get("rejected", False),
    )


@app.post("/translate/text", response_model=TextTranslationResponse)
async def translate_text(request: TextTranslationRequest):
    """Translate text between TSL and other languages."""
    dictionary = TSLDictionary()

    # Try as TSL label first
    result = dictionary.get_all_translations(request.text)
    if result:
        return TextTranslationResponse(
            input=request.text,
            tsl=result.get('tsl'),
            tunisian=result.get('tunisian'),
            french=result.get('french'),
            english=result.get('english'),
            arabic=result.get('arabic'),
            arabic_msa=result.get('arabic_msa'),
            category=result.get('category'),
        )

    # Try as Tunisian dialect (case-insensitive)
    tsl_label = dictionary.tunisian_to_tsl.get(request.text.lower())
    if tsl_label:
        result = dictionary.get_all_translations(tsl_label)
        return TextTranslationResponse(
            input=request.text,
            tsl=tsl_label,
            tunisian=request.text,
            french=result.get('french'),
            english=result.get('english'),
            arabic=result.get('arabic'),
            arabic_msa=result.get('arabic_msa'),
            category=result.get('category'),
        )

    return TextTranslationResponse(input=request.text)


class MultiTranslateRequest(BaseModel):
    """Translate text to multiple target languages."""
    text: str
    source_lang: str = "french"
    target_langs: List[str] = ["english", "arabic"]


@app.post("/translate/multi")
async def translate_multi(request: MultiTranslateRequest):
    """Translate arbitrary text to multiple target languages via offline Argos.

    source_lang: 'french', 'english', 'arabic', 'tunisian', 'spanish',
                  'german' etc. 'tunisian' is mapped to 'ar' for translation.
    """
    from src.translation.translator_backend import OfflineTranslator
    translator_be = OfflineTranslator()
    out = {}
    for lang in request.target_langs:
        out[lang] = translator_be.translate(request.text, request.source_lang, lang)
    return {
        "input": request.text,
        "source_lang": request.source_lang,
        "translations": out,
    }


@app.get("/translate/languages")
async def list_languages():
    """List languages available for offline translation."""
    from src.translation.translator_backend import OfflineTranslator
    translator_be = OfflineTranslator()
    return {"languages": translator_be.available_languages()}


@app.post("/translate/video")
async def translate_video(
    file: UploadFile = File(...),
    target_lang: str = "en",
    extra_langs: str = "en,fr,ar",
):
    """Process an uploaded video file and return assembled sentences with
    multi-language translations of each sentence.

    Uses StreamingRecognizer + SentenceBuilder under the hood for continuous
    recognition + phrase detection.
    """
    if translator is None:
        raise HTTPException(status_code=503, detail="Translator not initialized")

    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:  # 50 MB cap for video
        raise HTTPException(status_code=413, detail="Video too large (max 50 MB)")

    # Save to a temp file because cv2.VideoCapture needs a path
    import tempfile, os
    suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    try:
        extra = [l.strip() for l in extra_langs.split(",") if l.strip()]
        sentences = await run_in_threadpool(
            translator.translate_video_file,
            tmp_path,
            target_lang=target_lang,
            extra_langs=extra,
            sample_every_n_frames=3,
        )
        return {
            "sentences": sentences,
            "count": len(sentences),
            "target_lang": target_lang,
            "extra_langs": extra,
        }
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.get("/dictionary/labels")
async def get_labels():
    """Get all TSL labels in canonical order."""
    dictionary = TSLDictionary()
    return {"labels": dictionary.get_all_labels(), "count": len(dictionary)}


@app.get("/dictionary/words")
async def get_words():
    """Get all Tunisian dialect words."""
    dictionary = TSLDictionary()
    return {"words": dictionary.get_all_tunisian_words()}


@app.get("/dictionary/categories")
async def get_categories():
    """Get all categories and their labels."""
    dictionary = TSLDictionary()
    return {
        "categories": {
            cat: dictionary.get_labels_by_category(cat)
            for cat in ["Demandes", "Destinations", "Famille", "Jours", "Transport"]
        }
    }


@app.get("/dictionary/{label}")
async def get_translation(label: str):
    """Get translation for a TSL label."""
    dictionary = TSLDictionary()
    result = dictionary.get_all_translations(label)
    if not result:
        raise HTTPException(status_code=404, detail=f"Label '{label}' not found")
    return result


@app.get("/history")
async def get_history():
    """Get translation history (bounded)."""
    if translator is None:
        raise HTTPException(status_code=503, detail="Translator not initialized")
    return {"history": translator.get_history(), "count": len(translator.get_history())}


@app.delete("/history")
async def clear_history():
    """Clear translation history."""
    if translator is None:
        raise HTTPException(status_code=503, detail="Translator not initialized")
    translator.clear_history()
    return {"message": "History cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
