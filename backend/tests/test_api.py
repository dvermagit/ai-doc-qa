from io import BytesIO
import sys
import types

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.datastructures import Headers

from app.api import routes
from app.core.config import get_settings
from app.core.security import create_access_token
from app.main import app
from app.models.db import SessionLocal, get_db, init_db
from app.models.entities import ContentChunk
from app.schemas.api import ChatRequest, UserCreate, UserLogin
from app.services.rate_limit import rate_limit

from app.services.chunking import chunk_text
from app.services.embeddings import EmbeddingService, cosine_similarity
from app.services import extraction
from app.services.extraction import classify_upload
from app.services.rate_limit import _buckets


pytestmark = pytest.mark.asyncio


async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_signup_login_and_duplicate(client):
    signup = await client.post("/auth/signup", json={"email": "person@example.com", "password": "password123"})
    assert signup.status_code == 200
    duplicate = await client.post("/auth/signup", json={"email": "person@example.com", "password": "password123"})
    assert duplicate.status_code == 409
    login = await client.post("/auth/login", json={"email": "person@example.com", "password": "password123"})
    assert login.status_code == 200
    bad = await client.post("/auth/login", json={"email": "person@example.com", "password": "wrong"})
    assert bad.status_code == 401


async def test_protected_routes_require_token(client):
    response = await client.get("/files")
    assert response.status_code == 401


async def test_text_upload_list_summary_chat_and_history(client, auth_headers):
    files = {"file": ("notes.txt", b"Gemini powers document question answering with grounded citations.", "text/plain")}
    uploaded = await client.post("/upload", files=files, headers=auth_headers)
    assert uploaded.status_code == 200
    file_id = uploaded.json()["id"]
    assert uploaded.json()["kind"] == "text"

    listed = await client.get("/files", headers=auth_headers)
    assert listed.status_code == 200
    assert listed.json()[0]["filename"] == "notes.txt"

    detail = await client.get(f"/files/{file_id}", headers=auth_headers)
    assert detail.status_code == 200

    summary = await client.get(f"/files/{file_id}/summary", headers=auth_headers)
    assert summary.status_code == 200
    assert "Gemini" in summary.json()["summary"]

    chat = await client.post("/chat", json={"question": "What powers QA?", "file_id": file_id}, headers=auth_headers)
    assert chat.status_code == 200
    assert chat.json()["sources"]

    history = await client.get("/chat/history", headers=auth_headers)
    assert history.status_code == 200
    assert len(history.json()) == 2


async def test_overview_chat_falls_back_to_transcript_context(client, auth_headers, monkeypatch):
    files = {
        "file": (
            "overview.txt",
            b"The video discusses SEO strategies, UI UX improvements, LLMs, prompt engineering, and Cloud API usage.",
            "text/plain",
        )
    }
    uploaded = await client.post("/upload", files=files, headers=auth_headers)
    file_id = uploaded.json()["id"]

    async def not_found_answer(question, context):
        assert "File summary" in context
        assert "SEO strategies" in context
        return "I could not find the answer."

    monkeypatch.setattr(routes.llm_service, "answer", not_found_answer)
    response = await client.post(
        "/chat",
        json={"question": "what's inside this video?", "file_id": file_id},
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert "SEO strategies" in response.json()["answer"]


async def test_audio_upload_timestamps_and_stream(client, auth_headers):
    files = {"file": ("meeting.mp3", b"fake audio bytes", "audio/mpeg")}
    uploaded = await client.post("/upload", files=files, headers=auth_headers)
    assert uploaded.status_code == 200
    file_id = uploaded.json()["id"]

    timestamps = await client.post(
        "/timestamps/search",
        json={"topic": "processing", "file_id": file_id},
        headers=auth_headers,
    )
    assert timestamps.status_code == 200
    assert timestamps.json()[0]["start_time"] == 0.0

    async with client.stream("POST", "/chat/stream", json={"question": "What is ready?", "file_id": file_id}, headers=auth_headers) as response:
        assert response.status_code == 200
        body = await response.aread()
        assert b"data:" in body


async def test_upload_validation_and_missing_file(client, auth_headers):
    bad = await client.post(
        "/upload",
        files={"file": ("archive.zip", b"zip", "application/zip")},
        headers=auth_headers,
    )
    assert bad.status_code == 400 or bad.status_code == 500
    missing = await client.get("/files/999", headers=auth_headers)
    assert missing.status_code == 404


async def test_media_endpoint_and_auth_errors(client, auth_headers):
    uploaded = await client.post(
        "/upload",
        files={"file": ("clip.mp4", b"fake video bytes", "video/mp4")},
        headers=auth_headers,
    )
    file_id = uploaded.json()["id"]
    token = auth_headers["Authorization"].split(" ", 1)[1]

    media = await client.get(f"/files/{file_id}/media?token={token}")
    assert media.status_code == 200
    assert media.content == b"fake video bytes"

    bad_media = await client.get(f"/files/{file_id}/media?token=bad")
    assert bad_media.status_code == 401

    text_file = await client.post(
        "/upload",
        files={"file": ("notes.txt", b"plain notes", "text/plain")},
        headers=auth_headers,
    )
    not_media = await client.get(f"/files/{text_file.json()['id']}/media?token={token}")
    assert not_media.status_code == 404


async def test_rate_limit(client):
    _buckets.clear()
    for _ in range(3):
        response = await client.get("/health")
        assert response.status_code == 200


async def test_invalid_token_and_missing_user_paths(client):
    invalid = await client.get("/files", headers={"Authorization": "Bearer bad-token"})
    assert invalid.status_code == 401
    ghost = create_access_token("999")
    missing_user = await client.get("/files", headers={"Authorization": f"Bearer {ghost}"})
    assert missing_user.status_code == 401


async def test_embedding_and_chunking_helpers():
    chunks = chunk_text("alpha beta gamma " * 200, size=50, overlap=10)
    assert len(chunks) > 1
    service = EmbeddingService(dimensions=8)
    a = await service.embed("alpha beta")
    b = await service.embed("alpha beta")
    assert cosine_similarity(a, b) > 0.99


async def test_chat_helper_context_and_fallbacks():
    page_chunk = ContentChunk(
        id=1,
        file_id=1,
        owner_id=1,
        text="A page chunk about onboarding.",
        embedding=[1.0],
        page_number=3,
    )
    plain_chunk = ContentChunk(
        id=2,
        file_id=1,
        owner_id=1,
        text="A related non overview chunk.",
        embedding=[1.0],
    )
    context = routes._build_context([page_chunk, plain_chunk])
    assert "Page 3" in context
    assert "Source" in context
    assert routes._extractive_answer("Find onboarding", [plain_chunk]).startswith("I found related content")


async def test_classify_upload():
    assert classify_upload("application/pdf", "x.pdf") == "pdf"
    assert classify_upload("audio/wav", "x.wav") == "audio"
    assert classify_upload("video/mp4", "x.mp4") == "video"
    assert classify_upload("text/plain", "x.txt") == "text"
    with pytest.raises(ValueError):
        classify_upload("application/zip", "x.zip")


async def test_direct_branches_and_helpers(monkeypatch, tmp_path):
    settings = get_settings()
    old_limit = settings.max_upload_mb
    settings.max_upload_mb = 0
    try:
        too_large = await client_like_upload(b"x")
        assert too_large.status_code == 413
    finally:
        settings.max_upload_mb = old_limit

    sidecar_media = tmp_path / "talk.mp3"
    sidecar_media.write_bytes(b"audio")
    sidecar_media.with_suffix(".mp3.txt").write_text("topic one. topic two.", encoding="utf-8")
    assert routes.transcribe_media_chunks(sidecar_media)[0]["text"].startswith("topic")

    assert chunk_text("") == []
    assert cosine_similarity([], []) == 0.0


async def test_gemini_transcription_parser_and_client(monkeypatch, tmp_path):
    parsed = extraction._parse_transcript_segments(
        '```json\n[{"text":"hello world","start_time":2,"end_time":5}]\n```'
    )
    assert parsed == [{"text": "hello world", "start_time": 2.0, "end_time": 5.0}]

    plain = extraction._parse_transcript_segments("plain transcript text")
    assert plain[0]["text"] == "plain transcript text"

    invalid_payload = extraction._parse_transcript_segments('{"text":"not a list"}')
    assert invalid_payload == []

    media = tmp_path / "sample.mp4"
    media.write_bytes(b"video")
    settings = get_settings()
    old_key = settings.gemini_api_key
    settings.gemini_api_key = "fake-key"

    class Uploaded:
        name = "uploaded-file"

    class FakeModel:
        def __init__(self, name):
            self.name = name

        def generate_content(self, payload):
            class Response:
                text = '[{"text":"gemini segment","start_time":1,"end_time":3}]'

            return Response()

    class FakeGenai:
        GenerativeModel = FakeModel
        deleted = False

        @staticmethod
        def configure(api_key):
            assert api_key == "fake-key"

        @staticmethod
        def upload_file(path):
            assert path.endswith("sample.mp4")
            return Uploaded()

        @staticmethod
        def delete_file(name):
            FakeGenai.deleted = name == "uploaded-file"

    fake_google = types.ModuleType("google")
    fake_google.generativeai = FakeGenai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.generativeai", FakeGenai)
    try:
        segments = extraction._transcribe_with_gemini(media)
        assert segments[0]["text"] == "gemini segment"
        assert FakeGenai.deleted
    finally:
        settings.gemini_api_key = old_key


async def test_gemini_transcription_fallbacks_and_wait(monkeypatch, tmp_path):
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"video")
    settings = get_settings()
    old_key = settings.gemini_api_key
    settings.gemini_api_key = ""
    try:
        no_key = extraction.transcribe_media_chunks(media)
        assert "GEMINI_API_KEY" in no_key[0]["text"]

        settings.gemini_api_key = "fake-key"
        monkeypatch.setattr(extraction, "_transcribe_with_gemini", lambda path: [])
        empty = extraction.transcribe_media_chunks(media)
        assert "did not return usable" in empty[0]["text"]

        def fail(_):
            raise RuntimeError("bad media")

        monkeypatch.setattr(extraction, "_transcribe_with_gemini", fail)
        failed = extraction.transcribe_media_chunks(media)
        assert "bad media" in failed[0]["text"]
    finally:
        settings.gemini_api_key = old_key


async def test_gemini_file_wait_states(monkeypatch):
    class File:
        def __init__(self, state):
            self.state = state
            self.name = "file"

    class State:
        name = "PROCESSING"

    class Genai:
        calls = 0

        @staticmethod
        def get_file(name):
            Genai.calls += 1
            return File("ACTIVE")

    monkeypatch.setattr(extraction.time, "sleep", lambda _: None)
    assert extraction._gemini_file_state(File(State())) == "PROCESSING"
    assert extraction._wait_for_gemini_file(Genai, File("PROCESSING"), timeout_seconds=2).state == "ACTIVE"
    with pytest.raises(RuntimeError):
        extraction._wait_for_gemini_file(Genai, File("FAILED"), timeout_seconds=2)
    monkeypatch.setattr(extraction.time, "time", lambda: 100)
    with pytest.raises(RuntimeError):
        extraction._wait_for_gemini_file(Genai, File("PROCESSING"), timeout_seconds=0)


async def client_like_upload(content: bytes):
    with TestClient(app) as sync_client:
        signup = sync_client.post("/auth/signup", json={"email": "tiny@example.com", "password": "password123"})
        token = signup.json()["access_token"]
        return sync_client.post(
            "/upload",
            files={"file": ("tiny.txt", content, "text/plain")},
            headers={"Authorization": f"Bearer {token}"},
        )


async def test_startup_and_dependency_generator():
    await init_db()
    generator = get_db()
    session = await generator.__anext__()
    assert session is not None
    await generator.aclose()


async def test_direct_route_logic(monkeypatch):
    async with SessionLocal() as db:
        token = await routes.signup(UserCreate(email="direct@example.com", password="password123"), db)
        assert token.access_token
        with pytest.raises(HTTPException):
            await routes.signup(UserCreate(email="direct@example.com", password="password123"), db)

        login = await routes.login(UserLogin(email="direct@example.com", password="password123"), db)
        assert login.access_token
        with pytest.raises(HTTPException):
            await routes.login(UserLogin(email="direct@example.com", password="wrong"), db)

        user = await routes.get_current_user(token.access_token, db)
        upload = upload_file_obj("direct.txt", b"direct route content about revenue", "text/plain")
        uploaded = await routes.upload_file(upload, db, user)
        assert uploaded.filename == "direct.txt"
        assert await routes.list_files(db, user)
        assert (await routes.get_file(uploaded.id, db, user)).id == uploaded.id
        assert (await routes.get_summary(uploaded.id, db, user)).summary

        answer = await routes.chat(ChatRequest(question="What content?", file_id=uploaded.id), db, user)
        assert answer.sources
        stream = await routes.chat_stream(ChatRequest(question="What content?", file_id=uploaded.id), db, user)
        assert stream.media_type == "text/event-stream"

        with pytest.raises(HTTPException):
            await routes._owned_file(db, user.id, 9999)

        monkeypatch.setattr(routes, "extract_pdf_chunks", lambda path: [{"text": "pdf page text", "page_number": 1}])
        pdf = await routes.upload_file(upload_file_obj("paper.pdf", b"%PDF fake", "application/pdf"), db, user)
        assert pdf.kind == "pdf"


async def test_direct_rate_limit_branches():
    class Request:
        client = None

    settings = get_settings()
    old_limit = settings.rate_limit_per_minute
    settings.rate_limit_per_minute = 0
    _buckets.clear()
    try:
        with pytest.raises(HTTPException):
            await rate_limit(Request())
    finally:
        settings.rate_limit_per_minute = old_limit
        _buckets.clear()


def upload_file_obj(name: str, content: bytes, content_type: str):
    return routes.UploadFile(
        filename=name,
        file=BytesIO(content),
        headers=Headers({"content-type": content_type}),
    )
