from app.core.config import get_settings


class LLMService:
    async def answer(self, question: str, context: str) -> str:
        settings = get_settings()
        prompt = (
            "Answer using only the provided uploaded-file context. "
            "If the user asks for an overview, summary, or what the file is about, "
            "summarize the main topics found in the context. "
            "Only say you could not find it when the context is empty or unrelated.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )
        if settings.gemini_api_key:
            try:
                import google.generativeai as genai

                genai.configure(api_key=settings.gemini_api_key)
                model = genai.GenerativeModel(settings.gemini_model)
                response = model.generate_content(prompt)
                return response.text or ""
            except Exception:
                pass
        if not context.strip():
            return "I could not find relevant content in the uploaded files."
        return f"Based on the uploaded content: {context[:700]}"

    async def summarize(self, text: str) -> str:
        settings = get_settings()
        prompt = f"Summarize this uploaded content clearly and concisely:\n\n{text[:12000]}"
        if settings.gemini_api_key:
            try:
                import google.generativeai as genai

                genai.configure(api_key=settings.gemini_api_key)
                model = genai.GenerativeModel(settings.gemini_model)
                response = model.generate_content(prompt)
                return response.text or ""
            except Exception:
                pass
        words = text.split()
        return " ".join(words[:90]) + ("..." if len(words) > 90 else "")
