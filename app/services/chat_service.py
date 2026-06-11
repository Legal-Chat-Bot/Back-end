"""
chat_service: RAG 전체 흐름 지휘 (Service 계층)

역할:
- 흩어진 부품(vector / message_crud / prompt / service / hallucination)을
  순서대로 호출해서 최종 답변을 완성하는 "리모컨"
- 의수의 웹소켓 라우터가 이 함수 하나만 호출하면 됨
  (ai_answer = "..." 자리에 process_chat 연결)

흐름:
  embed → search → 거름망 → 이전대화 → 조립 → LLM → 환각검증 → 반환
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


async def process_chat(
    db: Session,
    user_id: str,
    session_id: str,
    question: str,
    context_mode: str = "general",   # 기본은 공용 검색. 파일 있으면 hybrid (추후 자동 판단)
    top_k: int = 5,
    history_limit: int = 10,
) -> ChatAnswerResponse:
    """
    질문 하나를 받아 RAG 파이프라인 전체를 실행하고 최종 응답 반환

    입력:
        db           : DB 세션
        user_id      : 질문한 유저
        session_id   : 채팅방
        question     : 사용자 질문
        context_mode : general(공용) / document(개인) / hybrid(둘 다)
        top_k        : 검색 결과 개수
        history_limit: 가져올 이전 대화 턴 수

    반환:
        ChatAnswerResponse (answer / verified / warnings / unverified_refs / sources)
    """

    # 2. (추후 추가) 이 세션에 업로드된 파일이 있으면 context_mode = "hybrid"
    #    지금은 Document 테이블 구조 확정 전이라 전달받은 값(기본 general) 그대로 사용

    # 3. Pinecone 검색 → Top-K
    search_results = search_pinecone(
        question=question,
        context_mode=context_mode,
        user_id=user_id,
        top_k=top_k,
    )

    # 4. 거름망 — 법률 질문인지 판별
    is_legal = is_legal_domain(search_results)
    if not is_legal:
        # 비법률 질문이면 LLM 호출 없이 안내 응답 반환
        return ChatAnswerResponse(
            answer="죄송합니다. 법률 관련 질문에만 답변할 수 있습니다.",
            verified=True,
            warnings=[],
            unverified_refs=[],
            sources=[],
        )

    # 5. 이전 대화 이력 가져오기 (최신순으로 반환됨 → format_history가 뒤집어 사용)
    history = get_recent_messages(
        db=db,
        session_id=session_id,
        limit=history_limit,
    )

    # 6. 프롬프트 조립 (시스템 + 검색결과 + 이전대화 + 질문)
    messages = assemble_messages(
        question=question,
        search_results=search_results,
        history=history,
    )

    # 7. LLM 호출 → 답변 문자열
    answer = await generate_answer(messages)

    # 8. 환각 검증 + 출처 포맷 → 최종 응답 조립
    response = build_response(answer, search_results)

    return response