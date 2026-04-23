import json
import re
import time
from pathlib import Path

from pypdf import PdfReader

from app.core.config import get_settings
from app.services.chunking import chunk_text


def extract_pdf_chunks(path: Path) -> list[dict]:
    reader = PdfReader(str(path))
    chunks: list[dict] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for chunk in chunk_text(text):
            chunks.append({"text": chunk, "page_number": index})
    return chunks


def transcribe_media_chunks(path: Path) -> list[dict]:
    sidecar = path.with_suffix(path.suffix + ".txt")
    if sidecar.exists():
        text = sidecar.read_text(encoding="utf-8")
        return _timestamped_chunks(text)

    if get_settings().gemini_api_key:
        try:
            gemini_chunks = _transcribe_with_gemini(path)
            if gemini_chunks:
                return gemini_chunks
            return _timestamped_chunks(
                "Gemini transcription completed, but it did not return usable transcript segments. "
                "Try re-uploading the file or using a clearer audio/video sample."
            )
        except RuntimeError as exc:
            return _timestamped_chunks(f"Gemini transcription failed: {exc}")

    text = (
        "Gemini transcription is not configured for this local run. Add GEMINI_API_KEY to .env, "
        "restart the backend, and re-upload this media file. "
        f"Uploaded media file {path.name} is stored and ready for processing."
    )
    return _timestamped_chunks(text)


def _timestamped_chunks(text: str) -> list[dict]:
    chunks = []
    for idx, chunk in enumerate(chunk_text(text, size=500, overlap=60)):
        start = float(idx * 30)
        chunks.append({"text": chunk, "start_time": start, "end_time": start + 30.0})
    return chunks


def _transcribe_with_gemini(path: Path) -> list[dict]:
    settings = get_settings()
    if not settings.gemini_api_key:
        return []
    uploaded = None
    try:
        import google.generativeai as genai

        genai.configure(api_key=settings.gemini_api_key)
        uploaded = genai.upload_file(str(path))
        uploaded = _wait_for_gemini_file(genai, uploaded)
        model = genai.GenerativeModel(settings.gemini_model)
        response = model.generate_content(
            [
                (
                    "Transcribe this audio/video into timestamped segments. Include spoken content "
                    "and any visible on-screen text that helps answer questions. If the speaker "
                    "states their name or the name appears on screen, include it. Do not infer a "
                    "person's identity from face or appearance. "
                    "Return only valid JSON as an array of objects with text, "
                    "start_time, and end_time fields. Times must be seconds."
                ),
                uploaded,
            ]
        )
        return _parse_transcript_segments(response.text or "")
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc
    finally:
        if uploaded is not None:
            try:
                genai.delete_file(uploaded.name)
            except Exception:
                pass


def _wait_for_gemini_file(genai, uploaded_file, timeout_seconds: int = 90):
    deadline = time.time() + timeout_seconds
    current = uploaded_file
    while time.time() < deadline:
        state = _gemini_file_state(current)
        if state in {"ACTIVE", "SUCCEEDED", "STATE_UNSPECIFIED", ""}:
            return current
        if state in {"FAILED", "ERROR"}:
            raise RuntimeError(f"Gemini file processing failed with state {state}")
        time.sleep(2)
        current = genai.get_file(current.name)
    raise RuntimeError("Gemini file processing timed out")


def _gemini_file_state(uploaded_file) -> str:
    state = getattr(uploaded_file, "state", "")
    name = getattr(state, "name", state)
    return str(name).split(".")[-1].upper()


def _parse_transcript_segments(raw_text: str) -> list[dict]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return _timestamped_chunks(cleaned)

    if not isinstance(payload, list):
        return []

    segments = []
    for item in payload:
        if not isinstance(item, dict) or not item.get("text"):
            continue
        start_time = float(item.get("start_time") or 0.0)
        end_time = float(item.get("end_time") or start_time + 30.0)
        segments.append({"text": str(item["text"]), "start_time": start_time, "end_time": end_time})
    return segments


def extract_plain_text_chunks(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [{"text": chunk} for chunk in chunk_text(text)]


def classify_upload(content_type: str, filename: str) -> str:
    lowered = filename.lower()
    if content_type == "application/pdf" or lowered.endswith(".pdf"):
        return "pdf"
    if content_type.startswith("audio/") or re.search(r"\.(mp3|wav|m4a)$", lowered):
        return "audio"
    if content_type.startswith("video/") or re.search(r"\.(mp4|mov|webm)$", lowered):
        return "video"
    if lowered.endswith(".txt"):
        return "text"
    raise ValueError("Unsupported file type")
