"""
FastAPI endpoints for TSL Translator.

Highlights:
  - CORS origins configurable via env (`TSL_CORS_ORIGINS`), explicit allow-list
    by default; credentials only enabled when origins are not "*".
  - `POST /llm/config` optionally gated by an admin token (`TSL_ADMIN_TOKEN`)
    with strict `base_url` validation (SSRF guard).
  - Per-client sessions (`X-Session-Id` header, one session per WebSocket) so
    concurrent clients never share history, mode or smoothing state.
  - Upload size limits enforced while streaming (never buffer past the cap).
  - Inference runs in a threadpool to avoid blocking the event loop; heavy
    offline-translation endpoints are plain `def` so Starlette runs them in
    its threadpool.
  - History is bounded per session (`HISTORY_MAX_LEN`).
  - Simple in-memory rate limiting on the LLM endpoints.
"""

import os
import json
import time
import uuid
import base64
import secrets
from contextlib import asynccontextmanager
from typing import Optional, Dict, List

try:  # optional dependency; .env loading is a convenience only
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

import numpy as np
import cv2

from src.translation.translator import TSLTranslator, get_shared_dictionary
from src.translation.translator_backend import get_default_translator
from src.languages.alphabet import TSLAlphabet
from src.config.settings import MAX_UPLOAD_BYTES


# Global translator instance (initialized on startup)
translator: Optional[TSLTranslator] = None

SESSION_HEADER = "X-Session-Id"
MAX_TEXT_CHARS = 4000          # cap for /llm/correct and text inputs
MAX_WS_MSG_BYTES = 6_000_000   # ~4.5 MB decoded JPEG + JSON overhead
VIDEO_MAX_BYTES = 50 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize translator on startup, cleanup on shutdown."""
    global translator
    translator = TSLTranslator()
    print("TSL Translator initialized")
    yield
    if translator and hasattr(translator, 'close'):
        await translator.close()


# Create FastAPI app
app = FastAPI(
    title="TSL Translator API",
    description="Tunisian Sign Language (W7CH dataset) to Dialect Translation API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — configurable via env, safe defaults
_default_origins = "http://localhost:8501,http://127.0.0.1:8501,http://localhost:3000,http://127.0.0.1:3000"
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


# ── Helpers ─────────────────────────────────────────────────────────────

def _session_id(request: Request) -> str:
    """Session taken from the X-Session-Id header (or the shared default)."""
    sid = (request.headers.get(SESSION_HEADER) or "").strip()
    return sid[:64] if sid else "default"


def _require_admin(request: Request) -> None:
    """Guard state-changing endpoints when TSL_ADMIN_TOKEN is configured."""
    token = os.environ.get("TSL_ADMIN_TOKEN", "")
    if not token:
        return  # auth disabled (local-dev default); SSRF guard still applies
    provided = request.headers.get("X-Admin-Token", "")
    if not secrets.compare_digest(provided.encode(), token.encode()):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Admin-Token")


_rate_hits: Dict[str, List[float]] = {}
_RATE_WINDOW_S = 60.0


def _rate_limit(request: Request, bucket: str, limit: int) -> None:
    """Minimal in-memory per-IP sliding-window rate limit."""
    ip = request.client.host if request.client else "unknown"
    key = f"{bucket}:{ip}"
    now = time.monotonic()
    hits = [t for t in _rate_hits.get(key, ()) if now - t < _RATE_WINDOW_S]
    if len(hits) >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({limit} requests/min for {bucket})",
        )
    hits.append(now)
    _rate_hits[key] = hits
    if len(_rate_hits) > 512:
        _rate_hits.clear()  # crude but bounded


async def _read_capped(file: UploadFile, cap: int) -> bytes:
    """Stream-read an upload, aborting as soon as the cap is exceeded."""
    chunks: List[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > cap:
            raise HTTPException(
                status_code=413,
                detail=f"File too large (max {cap // (1024 * 1024)} MB)",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _require_translator() -> TSLTranslator:
    if translator is None:
        raise HTTPException(status_code=503, detail="Translator not initialized")
    return translator


# Languages the sentence assembler + LLM coordinator actually support.
# Anything else would silently fall back to English — better to refuse it.
TARGET_LANG_ALIASES = {
    "en": "en", "english": "en",
    "fr": "fr", "french": "fr",
    "ar": "ar", "arabic": "ar", "arabic_msa": "ar", "msa": "ar",
    "tunisian": "tunisian", "tn": "tunisian", "derja": "tunisian",
}


def _normalize_target_lang(lang: str) -> str:
    key = (lang or "").strip().lower()
    if key not in TARGET_LANG_ALIASES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported target_lang '{lang}'. Supported: "
                + ", ".join(sorted(set(TARGET_LANG_ALIASES.values())))
            ),
        )
    return TARGET_LANG_ALIASES[key]


# ── Models ──────────────────────────────────────────────────────────────

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
    arabic_msa: Optional[str] = None
    category: Optional[str] = None
    target_lang: str = "french"


# ── Core endpoints ──────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "TSL Translator API",
        "version": app.version,
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    ready = translator is not None
    return {"status": "healthy" if ready else "starting", "ready": ready}


@app.post("/translate/frame", response_model=TranslationResponse)
async def translate_frame(request: Request, file: UploadFile = File(...)):
    """
    Translate a single frame (image).

    Accepts JPEG/PNG up to MAX_UPLOAD_BYTES (enforced while streaming).
    Inference runs in a threadpool so the event loop stays responsive.
    Pass an optional X-Session-Id header to isolate smoothing/history.
    """
    tr = _require_translator()
    sid = _session_id(request)

    contents = await _read_capped(file, MAX_UPLOAD_BYTES)

    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image format")

    # Run CPU-bound inference in a thread (don't block the loop)
    result = await run_in_threadpool(tr.translate_frame, frame, session_id=sid)

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return TranslationResponse(
        tsl=result.get("tsl"),
        tunisian=result.get("tunisian"),
        french=result.get("french"),
        english=result.get("english"),
        arabic=result.get("arabic"),
        arabic_msa=result.get("arabic_msa"),
        category=result.get("category"),
        confidence=result.get("confidence", 0.0),
        rejected=result.get("rejected", False),
    )


@app.post("/translate/text", response_model=TextTranslationResponse)
async def translate_text(request: TextTranslationRequest):
    """Translate text between TSL and other languages."""
    if len(request.text) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=413, detail=f"text too long (max {MAX_TEXT_CHARS} chars)")
    dictionary = get_shared_dictionary()

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
            target_lang=request.target_lang,
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
            target_lang=request.target_lang,
        )

    return TextTranslationResponse(input=request.text, target_lang=request.target_lang)


class MultiTranslateRequest(BaseModel):
    """Translate text to multiple target languages."""
    text: str
    source_lang: str = "french"
    target_langs: List[str] = ["english", "arabic"]


@app.post("/translate/multi")
def translate_multi(request: MultiTranslateRequest):
    """Translate arbitrary text to multiple target languages via offline Argos.

    source_lang: 'french', 'english', 'arabic', 'tunisian', 'spanish',
                  'german' etc. 'tunisian' is mapped to 'ar' for translation.

    Declared as `def` (not async): offline translation is CPU-bound, so
    Starlette runs it in its threadpool instead of blocking the event loop.
    """
    if len(request.text) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=413, detail=f"text too long (max {MAX_TEXT_CHARS} chars)")
    if len(request.target_langs) > 10:
        raise HTTPException(status_code=400, detail="Too many target languages (max 10)")

    translator_be = get_default_translator()
    out = {}
    for lang in request.target_langs:
        out[lang] = translator_be.translate(request.text, request.source_lang, lang)
    return {
        "input": request.text,
        "source_lang": request.source_lang,
        "translations": out,
    }


@app.get("/translate/languages")
def list_languages():
    """List languages available for offline translation.

    Declared as `def`: runs in the server threadpool.
    """
    return {"languages": get_default_translator().available_languages()}


@app.post("/translate/video")
async def translate_video(
    request: Request,
    file: UploadFile = File(...),
    target_lang: str = "en",
    extra_langs: str = "en,fr,ar",
    sample_rate: int = Query(4, ge=1, le=16, description="Process 1 frame every N"),
    llm: str = Query("1", description="Coordinate sentences with the LLM (1/0)"),
):
    """Process an uploaded video file and return assembled sentences with
    multi-language translations of each sentence.

    Uses StreamingRecognizer + SentenceBuilder under the hood for continuous
    recognition + phrase detection. Video processing uses a private session,
    so it never touches other clients' state. When the LLM is enabled, each
    sentence is additionally coordinated (natural wording in target_lang)
    and its translations refreshed from the coordinated text.
    """
    tr = _require_translator()
    target_lang = _normalize_target_lang(target_lang)

    contents = await _read_capped(file, VIDEO_MAX_BYTES)

    # Save to a temp file because cv2.VideoCapture needs a path
    import tempfile
    suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    try:
        extra = [l.strip() for l in extra_langs.split(",") if l.strip()][:8]
        sentences = await run_in_threadpool(
            tr.translate_video_file,
            tmp_path,
            target_lang=target_lang,
            extra_langs=extra,
            sample_every_n_frames=sample_rate,
        )
        use_llm = llm.strip().lower() in ("1", "true", "yes", "on")
        if sentences and use_llm:
            sentences = await tr.coordinate_sentences(
                sentences, target_lang=target_lang
            )
        return {
            "sentences": sentences,
            "count": len(sentences),
            "target_lang": target_lang,
            "extra_langs": extra,
            "llm": use_llm,
        }
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ── Dictionary ──────────────────────────────────────────────────────────

@app.get("/dictionary/labels")
async def get_labels():
    """Get all TSL labels in canonical order."""
    dictionary = get_shared_dictionary()
    return {"labels": dictionary.get_all_labels(), "count": len(dictionary)}


@app.get("/dictionary/words")
async def get_words():
    """Get all Tunisian dialect words."""
    dictionary = get_shared_dictionary()
    return {"words": dictionary.get_all_tunisian_words()}


@app.get("/dictionary/categories")
async def get_categories():
    """Get all categories and their labels."""
    dictionary = get_shared_dictionary()
    return {
        "categories": {
            cat: dictionary.get_labels_by_category(cat)
            for cat in ["Demandes", "Destinations", "Famille", "Jours", "Transport"]
        }
    }


@app.get("/dictionary/{label}")
async def get_translation(label: str):
    """Get translation for a TSL label."""
    dictionary = get_shared_dictionary()
    result = dictionary.get_all_translations(label)
    if not result:
        raise HTTPException(status_code=404, detail=f"Label '{label}' not found")
    return result


# ── History (per session) ───────────────────────────────────────────────

@app.get("/history")
async def get_history(request: Request):
    """Get translation history for this client session (bounded)."""
    tr = _require_translator()
    history = tr.get_history(_session_id(request))
    return {"history": history, "count": len(history)}


@app.delete("/history")
async def clear_history(request: Request):
    """Clear translation history for this client session."""
    tr = _require_translator()
    tr.clear_history(_session_id(request))
    return {"message": "History cleared"}


# ── Alphabet / Finger Spelling ──────────────────────────────────────────

@app.get("/alphabet/labels")
async def get_alphabet_labels():
    """Get all TSL alphabet labels for finger spelling."""
    alphabet = TSLAlphabet()
    labels = list(alphabet.sign_to_index.keys())
    return {"labels": labels, "count": len(labels)}


@app.get("/alphabet/{label}")
async def get_alphabet_translation(label: str):
    """Get translation for an alphabet label."""
    alphabet = TSLAlphabet()
    result = alphabet.get_translations(label)
    if not result:
        raise HTTPException(status_code=404, detail=f"Alphabet label '{label}' not found")
    return result


# ── Inference settings ──────────────────────────────────────────────────

class InferenceSettingsRequest(BaseModel):
    confidence_threshold: Optional[float] = None
    smoothing_window: Optional[int] = None


@app.get("/settings/inference")
async def get_inference_settings():
    """Get current global inference settings."""
    return _require_translator().get_inference_settings()


@app.post("/settings/inference")
async def set_inference_settings(request: InferenceSettingsRequest):
    """Update confidence threshold and/or smoothing window (validated)."""
    tr = _require_translator()
    try:
        return tr.set_inference_settings(
            confidence_threshold=request.confidence_threshold,
            smoothing_window=request.smoothing_window,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── LLM Configuration ───────────────────────────────────────────────────

class LLMConfigRequest(BaseModel):
    """Partial update: every field is optional.

    ``None`` (field omitted) means "keep the current value" — this is what
    stops a partial payload from resetting the provider to the OpenRouter
    default or wiping a stored API key. ``api_key`` sent as an empty string
    explicitly clears the key.
    """
    provider: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None


@app.get("/llm/config")
async def get_llm_config():
    """Get current LLM corrector configuration (never exposes the key)."""
    return _require_translator().get_llm_config()


@app.post("/llm/config")
async def set_llm_config(request: Request, body: LLMConfigRequest):
    """Configure the LLM corrector (provider, API key, model).

    Hardened:
      * optional X-Admin-Token auth when TSL_ADMIN_TOKEN is set;
      * per-IP rate limit;
      * LLMConfig.validate() rejects arbitrary/unsafe base_url values
        (SSRF guard) — see LLM_ALLOW_CUSTOM_BASE_URL / LLM_ALLOWED_HOSTS.
    """
    tr = _require_translator()
    _require_admin(request)
    _rate_limit(request, "llm_config", limit=10)

    if len(body.api_key or "") > 512 or len(body.base_url or "") > 512 or len(body.model or "") > 256:
        raise HTTPException(status_code=413, detail="Configuration field too long")

    try:
        tr.configure_llm(
            provider=body.provider,
            api_key=body.api_key,
            model=body.model,
            base_url=body.base_url,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"message": "LLM configured", "config": tr.get_llm_config()}


@app.post("/llm/correct")
async def correct_with_llm(request: Request, body: TextTranslationRequest):
    """Correct a text using the configured LLM (rate-limited, size-capped)."""
    tr = _require_translator()
    _rate_limit(request, "llm_correct", limit=60)
    text = body.text[:MAX_TEXT_CHARS]
    corrected = await tr.correct_with_llm(text)
    return {"raw": text, "corrected": corrected}


# ── Mode Controller (per session) ───────────────────────────────────────

@app.get("/mode")
async def get_mode(request: Request):
    """Get current translation mode (sign/spell) for this session."""
    return _require_translator().get_mode_state(_session_id(request))


@app.post("/mode/toggle")
async def toggle_mode(request: Request):
    """Toggle between sign and spell mode for this session."""
    tr = _require_translator()
    sid = _session_id(request)
    new_mode = tr.toggle_mode(sid)
    return {"mode": new_mode, "state": tr.get_mode_state(sid)}


@app.post("/spell/flush")
async def flush_spell(request: Request):
    """Flush the spell assembler and return the assembled sentence."""
    sentence = _require_translator().flush_spell(_session_id(request))
    return {"sentence": sentence}


# ── WebSocket for real-time camera ──────────────────────────────────────

@app.websocket("/ws/camera")
async def websocket_camera(websocket: WebSocket):
    """
    WebSocket endpoint for real-time camera translation.

    Each connection gets its own session (isolated history/mode/smoothing).

    Client sends: base64-encoded JPEG frames
    Server sends: translation results as JSON

    Protocol:
      - Client connects
      - Client sends JSON: {"frame": "<base64>"}
      - Server responds: {"type": "translation", "data": {...}}
        or {"type": "spell", "data": {...}} / {"type": "mode_changed", ...}
      - Client can send: {"action": "toggle_mode"}
      - Client can send: {"action": "flush_spell"}
      - Client can send: {"action": "clear_history"}
      - Client can send: {"action": "correct", "text": "..."}
    """
    await websocket.accept()
    tr = translator
    if tr is None:
        await websocket.send_json({"type": "error", "message": "Translator not initialized"})
        await websocket.close()
        return

    session_id = uuid.uuid4().hex
    try:
        while True:
            raw = await websocket.receive_text()
            if len(raw) > MAX_WS_MSG_BYTES:
                await websocket.send_json({"type": "error", "message": "Message too large"})
                continue
            try:
                msg = json.loads(raw)
            except (ValueError, TypeError):
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue
            if not isinstance(msg, dict):
                await websocket.send_json({"type": "error", "message": "Invalid message"})
                continue

            action = msg.get("action")

            if action == "toggle_mode":
                await run_in_threadpool(tr.toggle_mode, session_id)
                await websocket.send_json({
                    "type": "mode_changed",
                    "data": tr.get_mode_state(session_id),
                })
                continue

            if action == "flush_spell":
                sentence = await run_in_threadpool(tr.flush_spell, session_id)
                await websocket.send_json({
                    "type": "spell_flushed",
                    "data": {"sentence": sentence},
                })
                continue

            if action == "clear_history":
                tr.clear_history(session_id)
                await websocket.send_json({"type": "history_cleared"})
                continue

            if action == "correct":
                text = str(msg.get("text", ""))[:MAX_TEXT_CHARS]
                corrected = await tr.correct_with_llm(text)
                await websocket.send_json({
                    "type": "corrected",
                    "data": {"raw": text, "corrected": corrected},
                })
                continue

            # Default: process frame (multi-mode: sign + spell + mode control)
            frame_b64 = msg.get("frame")
            if not frame_b64:
                await websocket.send_json({"type": "error", "message": "No frame provided"})
                continue
            if not isinstance(frame_b64, str) or len(frame_b64) > MAX_WS_MSG_BYTES:
                await websocket.send_json({"type": "error", "message": "Frame too large"})
                continue

            # Decode base64 frame
            try:
                frame_bytes = base64.b64decode(frame_b64, validate=False)
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    await websocket.send_json({"type": "error", "message": "Invalid frame"})
                    continue
            except Exception:
                await websocket.send_json({"type": "error", "message": "Failed to decode frame"})
                continue

            # Run inference in threadpool (session-scoped state)
            result = await run_in_threadpool(
                tr.translate_frame_multi_mode, frame, session_id=session_id
            )

            mtype = result.get("type")
            if mtype == "mode_change":
                await websocket.send_json({
                    "type": "mode_changed",
                    "data": tr.get_mode_state(session_id),
                    "sign": result.get("sign"),
                })
            elif mtype == "spell":
                await websocket.send_json({"type": "spell", "data": result})
            elif mtype == "error":
                await websocket.send_json({
                    "type": "error",
                    "message": result.get("error", "Unknown error"),
                })
            else:
                await websocket.send_json({"type": "translation", "data": result})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            tr.delete_session(session_id)
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
