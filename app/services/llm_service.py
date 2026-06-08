import httpx
from app.core.config import settings

async def generate_answer(messages : list[dict]) -> str:
    """
    Ollama에 메시지 리스트를 보내고 답변 문자열을 반환
    messages 형식 : [{"role" : "system", "content":"..."}, {"role":"user", "content":"..."}]
    """
    payload = {
        "model" : settings.RAG_MODEL,
        "messages" : messages,
        "stream" : False,
        "options" : {
            "temperature" : 0.3,
            "top_p" : 0.9,
            "repeat_penalty" : 1.1,
        }
    } 

    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            f"{settings.OLLAMA_BASE_URL}/api/chat",
            json = payload
        )
        response.raise_for_status()
        data = response.json()

        return data["message"]["content"]