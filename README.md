# AI-Powered Document & Multimedia Q&A

Full-stack FastAPI and React application for uploading PDFs, audio, and video, then asking grounded AI questions with summaries, citations, timestamps, and media playback.

## Features

- JWT signup and login
- Upload PDFs, text files, audio, and video
- PDF text extraction with page metadata
- Audio/video transcript chunking with timestamp metadata
- Gemini-powered answers and summaries via `GEMINI_API_KEY`
- Deterministic local embedding fallback for tests and development without secrets
- Semantic retrieval over stored chunks
- Timestamp search and Play buttons that seek media to the relevant moment
- FastAPI Swagger docs at `http://localhost:8000/docs`
- Pytest suite configured for 95% coverage
- Docker, Docker Compose, and GitHub Actions CI

## Quick Start

1. Create your environment file:

```bash
cp .env.example .env
```

2. Set `GEMINI_API_KEY` and `JWT_SECRET` in `.env`.

3. Start the app:

```bash
docker compose up --build
```

4. Open:

- Frontend: `http://localhost:3000`
- Backend API docs: `http://localhost:8000/docs`

## Local Backend Development

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Run tests:

```bash
cd backend
pytest
```

## Local Frontend Development

Use Node.js 20 or newer for the patched Vite toolchain. If you use `nvm`:

```bash
nvm install
nvm use
```

```bash
cd frontend
npm install
npm run dev
```

The frontend reads `VITE_API_URL`; if unset, it uses `http://localhost:8000`.

## Notes

- Secrets are loaded from environment variables and are never hardcoded.
- Media transcription uses Gemini when `GEMINI_API_KEY` is set. If no key is configured or the Gemini media request fails, the app uses a deterministic fallback so uploads and tests still run.
- The vector layer uses local embeddings by default and Gemini embeddings when `GEMINI_API_KEY` is available. Pinecone or FAISS can be added behind the same retrieval boundary.
