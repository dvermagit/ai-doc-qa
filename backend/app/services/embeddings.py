import hashlib
import math

from app.core.config import get_settings


class EmbeddingService:
    def __init__(self, dimensions: int = 64):
        self.dimensions = dimensions

    async def embed(self, text: str) -> list[float]:
        settings = get_settings()
        if settings.gemini_api_key:
            try:
                import google.generativeai as genai

                genai.configure(api_key=settings.gemini_api_key)
                result = genai.embed_content(
                    model=settings.gemini_embedding_model,
                    content=text,
                    task_type="retrieval_document",
                )
                return list(result["embedding"])
            except Exception:
                pass
        return self._hash_embedding(text)

    def _hash_embedding(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "big") % self.dimensions
            sign = 1 if digest[2] % 2 == 0 else -1
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    length = min(len(a), len(b))
    return sum(a[i] * b[i] for i in range(length))
