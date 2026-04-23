from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.config import get_settings
from app.core.security import create_access_token, decode_access_token, hash_password, verify_password
from app.models.db import get_db
from app.models.entities import ChatMessage, ContentChunk, UploadedFile, User
from app.schemas.api import (
    ChatRequest,
    ChatResponse,
    FileOut,
    SourceOut,
    SummaryOut,
    TimestampOut,
    TimestampSearchRequest,
    TokenResponse,
    UserCreate,
    UserLogin,
)
from app.services.embeddings import EmbeddingService, cosine_similarity
from app.services.extraction import classify_upload, extract_pdf_chunks, extract_plain_text_chunks, transcribe_media_chunks
from app.services.llm import LLMService


router = APIRouter()
embedding_service = EmbeddingService()
llm_service = LLMService()


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.post("/auth/signup", response_model=TokenResponse)
async def signup(payload: UserCreate, db: AsyncSession = Depends(get_db)) -> TokenResponse:
    existing = await db.execute(select(User).where(User.email == payload.email.lower()))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")
    user = User(email=payload.email.lower(), hashed_password=hash_password(payload.password))
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return TokenResponse(access_token=create_access_token(str(user.id)))


@router.post("/auth/login", response_model=TokenResponse)
async def login(payload: UserLogin, db: AsyncSession = Depends(get_db)) -> TokenResponse:
    result = await db.execute(select(User).where(User.email == payload.email.lower()))
    user = result.scalar_one_or_none()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return TokenResponse(access_token=create_access_token(str(user.id)))


@router.post("/upload", response_model=FileOut)
async def upload_file(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> FileOut:
    settings = get_settings()
    try:
        kind = classify_upload(file.content_type or "", file.filename or "")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    content = await file.read()
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid4().hex}_{Path(file.filename or 'upload').name}"
    path = upload_dir / safe_name
    path.write_bytes(content)

    if kind == "pdf":
        extracted = extract_pdf_chunks(path)
    elif kind in {"audio", "video"}:
        extracted = transcribe_media_chunks(path)
    else:
        extracted = extract_plain_text_chunks(path)

    summary = await llm_service.summarize(" ".join(item["text"] for item in extracted))
    uploaded = UploadedFile(
        owner_id=user.id,
        filename=file.filename or safe_name,
        content_type=file.content_type or "application/octet-stream",
        storage_path=str(path),
        kind=kind,
        summary=summary,
    )
    db.add(uploaded)
    await db.flush()
    for item in extracted:
        db.add(
            ContentChunk(
                file_id=uploaded.id,
                owner_id=user.id,
                text=item["text"],
                embedding=await embedding_service.embed(item["text"]),
                page_number=item.get("page_number"),
                start_time=item.get("start_time"),
                end_time=item.get("end_time"),
            )
        )
    await db.commit()
    await db.refresh(uploaded)
    return FileOut(
        id=uploaded.id,
        filename=uploaded.filename,
        content_type=uploaded.content_type,
        kind=uploaded.kind,
        summary=uploaded.summary,
    )


@router.get("/files", response_model=list[FileOut])
async def list_files(db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)) -> list[FileOut]:
    result = await db.execute(select(UploadedFile).where(UploadedFile.owner_id == user.id).order_by(UploadedFile.id.desc()))
    return [
        FileOut(id=f.id, filename=f.filename, content_type=f.content_type, kind=f.kind, summary=f.summary)
        for f in result.scalars().all()
    ]


@router.get("/files/{file_id}", response_model=FileOut)
async def get_file(file_id: int, db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)) -> FileOut:
    file = await _owned_file(db, user.id, file_id)
    return FileOut(id=file.id, filename=file.filename, content_type=file.content_type, kind=file.kind, summary=file.summary)


@router.get("/files/{file_id}/summary", response_model=SummaryOut)
async def get_summary(file_id: int, db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)) -> SummaryOut:
    file = await _owned_file(db, user.id, file_id)
    return SummaryOut(file_id=file.id, summary=file.summary)


@router.get("/files/{file_id}/media")
async def get_media(
    file_id: int,
    token: str | None = Query(None),
    authorization: str | None = Header(None),
    db: AsyncSession = Depends(get_db),
) -> FileResponse:
    bearer = authorization.removeprefix("Bearer ").strip() if authorization else None
    subject = decode_access_token(token or bearer or "")
    if not subject:
        raise HTTPException(status_code=401, detail="Invalid token")
    user_id = int(subject)
    file = await _owned_file(db, user_id, file_id)
    if file.kind not in {"audio", "video"}:
        raise HTTPException(status_code=404, detail="Media not found")
    return FileResponse(file.storage_path, media_type=file.content_type, filename=file.filename)


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)) -> ChatResponse:
    file = await _owned_file(db, user.id, payload.file_id) if payload.file_id else None
    if payload.file_id and _is_overview_question(payload.question):
        chunks = await _file_chunks(db, user.id, payload.file_id, limit=20)
    else:
        chunks = await _search_chunks(db, user.id, payload.question, payload.file_id, limit=8)

    context = _build_context(chunks, file.summary if file else "")
    answer = await llm_service.answer(payload.question, context)
    if chunks and _looks_like_not_found(answer):
        answer = _extractive_answer(payload.question, chunks, file.summary if file else "")
    sources = [_source_from_chunk(chunk) for chunk in chunks]
    db.add(ChatMessage(owner_id=user.id, file_id=payload.file_id, role="user", content=payload.question, metadata_json={}))
    db.add(ChatMessage(owner_id=user.id, file_id=payload.file_id, role="assistant", content=answer, metadata_json={"sources": [s.model_dump() for s in sources]}))
    await db.commit()
    return ChatResponse(answer=answer, sources=sources)


@router.post("/chat/stream")
async def chat_stream(payload: ChatRequest, db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    response = await chat(payload, db, user)

    async def event_stream():
        for word in response.answer.split():
            yield f"data: {word} \n\n"
        yield "event: sources\ndata: done\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/chat/history")
async def chat_history(db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)) -> list[dict]:
    result = await db.execute(select(ChatMessage).where(ChatMessage.owner_id == user.id).order_by(ChatMessage.id.desc()).limit(50))
    return [
        {"id": msg.id, "role": msg.role, "content": msg.content, "metadata": msg.metadata_json, "file_id": msg.file_id}
        for msg in reversed(result.scalars().all())
    ]


@router.post("/timestamps/search", response_model=list[TimestampOut])
async def timestamp_search(
    payload: TimestampSearchRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[TimestampOut]:
    chunks = await _search_chunks(db, user.id, payload.topic, payload.file_id, limit=5, media_only=True)
    return [
        TimestampOut(
            file_id=chunk.file_id,
            chunk_id=chunk.id,
            text=chunk.text,
            start_time=chunk.start_time or 0.0,
            end_time=chunk.end_time or 0.0,
        )
        for chunk in chunks
        if chunk.start_time is not None and chunk.end_time is not None
    ]


async def _owned_file(db: AsyncSession, user_id: int, file_id: int) -> UploadedFile:
    result = await db.execute(select(UploadedFile).where(UploadedFile.id == file_id, UploadedFile.owner_id == user_id))
    file = result.scalar_one_or_none()
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    return file


async def _search_chunks(
    db: AsyncSession,
    user_id: int,
    query: str,
    file_id: int | None,
    limit: int,
    media_only: bool = False,
) -> list[ContentChunk]:
    query_embedding = await embedding_service.embed(query)
    statement = select(ContentChunk).where(ContentChunk.owner_id == user_id)
    if file_id:
        statement = statement.where(ContentChunk.file_id == file_id)
    if media_only:
        statement = statement.where(ContentChunk.start_time.is_not(None))
    result = await db.execute(statement)
    chunks = result.scalars().all()
    return sorted(chunks, key=lambda chunk: cosine_similarity(query_embedding, chunk.embedding), reverse=True)[:limit]


async def _file_chunks(db: AsyncSession, user_id: int, file_id: int, limit: int) -> list[ContentChunk]:
    result = await db.execute(
        select(ContentChunk).where(ContentChunk.owner_id == user_id, ContentChunk.file_id == file_id)
    )
    chunks = result.scalars().all()
    return sorted(chunks, key=_chunk_order)[:limit]


def _chunk_order(chunk: ContentChunk) -> tuple:
    marker = chunk.start_time if chunk.start_time is not None else chunk.page_number
    return (marker is None, marker or 0, chunk.id)


def _build_context(chunks: list[ContentChunk], summary: str = "") -> str:
    parts = []
    if summary:
        parts.append(f"File summary:\n{summary}")
    for chunk in chunks:
        label = "Source"
        if chunk.page_number is not None:
            label = f"Page {chunk.page_number}"
        elif chunk.start_time is not None:
            label = f"{round(chunk.start_time, 2)}s-{round(chunk.end_time or chunk.start_time, 2)}s"
        parts.append(f"{label}:\n{chunk.text}")
    return "\n\n".join(parts)


def _is_overview_question(question: str) -> bool:
    normalized = question.lower().strip()
    phrases = (
        "what's inside",
        "whats inside",
        "what is inside",
        "what's in",
        "what is in",
        "what is this",
        "what's this",
        "what is the video",
        "what's the video",
        "what is speaker talking",
        "what is the speaker talking",
        "what is this about",
        "summarize",
        "summary",
        "overview",
    )
    return any(phrase in normalized for phrase in phrases)


def _looks_like_not_found(answer: str) -> bool:
    normalized = answer.lower()
    return "could not find" in normalized or "couldn't find" in normalized or "not found" in normalized


def _extractive_answer(question: str, chunks: list[ContentChunk], summary: str = "") -> str:
    if _is_overview_question(question):
        text = summary or " ".join(chunk.text for chunk in chunks[:8])
        return f"The video appears to discuss: {_compact_text(text, 700)}"
    text = " ".join(chunk.text for chunk in chunks[:3])
    return f"I found related content in the upload: {_compact_text(text, 500)}"


def _compact_text(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    return compact[:limit].rstrip() + ("..." if len(compact) > limit else "")


def _source_from_chunk(chunk: ContentChunk) -> SourceOut:
    return SourceOut(
        file_id=chunk.file_id,
        chunk_id=chunk.id,
        text=chunk.text[:300],
        page_number=chunk.page_number,
        start_time=chunk.start_time,
        end_time=chunk.end_time,
    )
