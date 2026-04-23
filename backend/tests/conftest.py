import os

os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test.db"
os.environ["GEMINI_API_KEY"] = ""
os.environ["JWT_SECRET"] = "test-secret"
os.environ["RATE_LIMIT_PER_MINUTE"] = "10000"
os.environ["UPLOAD_DIR"] = "test_uploads"

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.models.db import Base, engine


@pytest.fixture(autouse=True)
async def reset_database():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def auth_headers(client):
    response = await client.post("/auth/signup", json={"email": "user@example.com", "password": "password123"})
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
