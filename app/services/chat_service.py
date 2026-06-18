"""
chat_service: RAG 전체 흐름 지휘 (Service 계층)

역할:
- 흩어진 부품(vector / message_crud / prompt / service / hallucination)을
  순서대로 호출해서 최종 답변을 완성하는 "리모컨"
- 웹소켓 라우터가 이 함수 하나만 호출하면 됨

흐름:
  이전대화 → (context_mode 자동판단) → (검색쿼리 맥락보강)
  → search → 거름망 → 조립 → LLM → 환각검증 → 반환
"""

from sqlalchemy.orm import Session
from app.services.vector_service import (
    search_pinecone,
    is_legal_domain,
)
from app.crud.message_crud import get_recent_messages
from app.services.prompt_service import assemble_messages
from app.services.service import generate_answer
from app.services.hallucination_service import build_response
from app.schemas.chat.response import ChatAnswerResponse
from app.db.models.document import Document, Status


async def process_chat(
    db: Session,
    user_id: str,
    session_id: str,
    question: str,
    context_mode: str = "general",
    top_k: int = 5,
    history_limit: int = 10,
    history: list[dict] | None = None,
) -> ChatAnswerResponse:
    """
    질문 하나를 받아 RAG 파이프라인 전체를 실행하고 최종 응답 반환

    입력:
        db           : DB 세션
        user_id      : 질문한 유저
        session_id   : 채팅방
        question     : 사용자 질문
        context_mode : general 기본값, 파일 있으면 자동으로 hybrid 전환
        top_k        : 검색 결과 개수
        history_limit: 가져올 이전 대화 턴 수
        history      : 직접 주입 시 DB 조회 건너뜀 (테스트용)

    반환:
        ChatAnswerResponse (answer / verified / warnings / unverified_refs / sources)
    """

    # 0. context_mode 자동 판단
    #    이 session_id에 READY 파일 있으면 → hybrid
    #    없으면 → general 유지
    #    db=None이면 테스트 모드라 건너뜀
    if db is not None and context_mode == "general":
        has_file = db.query(Document).filter(
            Document.session_id == session_id,
            Document.status == Status.READY,
        ).first() is not None
        if has_file:
            context_mode = "hybrid"

    # print(f"1:{context_mode}")

    # 1. 이전 대화 이력 가져오기
    #    최신순 반환 → history[0]이 직전 질문
    #    history 직접 주입 시 DB 조회 건너뜀 (테스트용)
    if history is None:
        history = get_recent_messages(
            db=db,
            session_id=session_id,
            limit=history_limit,
        )

    # print(f"2:{history}")

    # 2. 검색용 쿼리 맥락 보강 (방법 A)
    #    지시어 후속질문은 직전 질문을 앞에 붙여 맥락 살림
    #    프롬프트에 들어가는 question은 원본 유지
    search_query = question
    if history:
        last_question = history[0].get("question", "")
        if last_question:
            search_query = f"{last_question} {question}"

    # print(f"3:{history}")

    # 3. Pinecone 검색 → Top-K
    search_results = search_pinecone(
        question=search_query,
        context_mode=context_mode,
        user_id=user_id,
        top_k=top_k,
    )

    # print(f"4:{search_results}")

    # 4. 거름망 — 법률 질문인지 판별
    is_legal = is_legal_domain(search_results)
    if not is_legal:
        return ChatAnswerResponse(
            answer="죄송합니다. 법률 관련 질문에만 답변할 수 있습니다.",
            verified=True,
            warnings=[],
            unverified_refs=[],
            sources=[],
        )
    
    # print(f"5:{is_legal}")

    # 5. 프롬프트 조립
    messages = assemble_messages(
        question=question,
        search_results=search_results,
        history=history,
    )

    # print(f"6:{messages}")

    # 6. LLM 호출
    answer = await generate_answer(messages)
    # print(f"7:{answer}")

    # 7. 환각 검증 + 출처 포맷 → 최종 응답
    response = build_response(answer, search_results)
    # print(f"8:{response}")

    return response