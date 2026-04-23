from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class FileOut(BaseModel):
    id: int
    filename: str
    content_type: str
    kind: str
    summary: str


class SourceOut(BaseModel):
    file_id: int
    chunk_id: int
    text: str
    page_number: int | None = None
    start_time: float | None = None
    end_time: float | None = None


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    file_id: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceOut]


class TimestampSearchRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=1000)
    file_id: int | None = None


class TimestampOut(BaseModel):
    file_id: int
    chunk_id: int
    text: str
    start_time: float
    end_time: float


class SummaryOut(BaseModel):
    file_id: int
    summary: str
