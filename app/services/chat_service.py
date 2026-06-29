"""
RAG 파이프라인 전체 흐름을 관리한다.
"""

from sqlalchemy.orm import Session

from app.services.vector_service import (
    LEGAL_MIN_SCORE,
    is_legal_domain,
    search_pinecone,
)
from app.crud.message_crud import get_recent_messages
from app.services.prompt_service import assemble_messages
from app.services.service import generate_answer
from app.services.hallucination_service import build_response
from app.schemas.chat.response import ChatAnswerResponse
from app.db.models.document import Document, Status


HARD_REJECT_SCORE = 1.3
CONTEXT_SEARCH_MAX = 2.0


def build_contextual_query(
    question: str,
    history: list[dict],
) -> str:
    """후속 질문이면 직전 질문 하나를 검색어에 붙인다."""

    if not history:
        return question

    previous_question = (
        history[0].get("question") or ""
    ).strip()

    if previous_question and previous_question != question:
        return f"{previous_question} {question}"

    return question


def get_score(result: dict) -> float:
    """검색 결과의 점수를 안전하게 반환한다."""

    try:
        return float(result.get("score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def select_llm_search_results(
    search_results: list[dict],
    context_mode: str,
    document_k: int = 2,
    public_k: int = 3,
) -> list[dict]:
    """
    파일 업로드 세션에서는:
    - 사용자 문서 최대 2개
    - 공용 법률 최대 3개

    일반 세션에서는 점수순 상위 5개를 사용한다.
    """

    total_k = document_k + public_k

    if context_mode != "hybrid":
        return search_results[:total_k]

    document_results = [
        result
        for result in search_results
        if result.get("source_type") == "document"
    ]

    public_results = [
        result
        for result in search_results
        if result.get("source_type") == "legal_vector"
    ]

    document_results.sort(
        key=get_score,
        reverse=True,
    )
    public_results.sort(
        key=get_score,
        reverse=True,
    )

    selected = (
        document_results[:document_k]
        + public_results[:public_k]
    )

    selected.sort(
        key=get_score,
        reverse=True,
    )

    return selected


async def process_chat(
    db: Session,
    user_id: str,
    session_id: str,
    question: str,
    context_mode: str = "general",
    top_k: int = 20,
    history_limit: int = 10,
    history: list[dict] | None = None,
) -> ChatAnswerResponse:
    """질문 하나를 받아 검색부터 답변 생성까지 실행한다."""

    # 파일이 있는 채팅방은 Hybrid 검색으로 전환한다.
    if db is not None and context_mode == "general":
        has_file = (
            db.query(Document)
            .filter(
                Document.session_id == session_id,
                Document.status.in_(
                    [
                        Status.READY,
                        Status.EMBEDDED,
                    ]
                ),
            )
            .first()
            is not None
        )

        if has_file:
            context_mode = "hybrid"

    # 완료된 이전 대화를 조회한다.
    if history is None:
        history = get_recent_messages(
            db=db,
            session_id=session_id,
            limit=history_limit,
        )

    # 현재 질문만으로 먼저 검색한다.
    standalone_results = search_pinecone(
        question=question,
        context_mode=context_mode,
        user_id=user_id,
        top_k=top_k,
        db=db,
    )

    standalone_top1 = (
        get_score(standalone_results[0])
        if standalone_results
        else 0.0
    )

    search_results = standalone_results

    # 점수가 애매하고 이전 대화가 있으면 맥락 검색을 수행한다.
    if (
        standalone_top1 <= CONTEXT_SEARCH_MAX
        and history
    ):
        contextual_query = build_contextual_query(
            question=question,
            history=history,
        )

        contextual_results = search_pinecone(
            question=contextual_query,
            context_mode=context_mode,
            user_id=user_id,
            top_k=top_k,
            db=db,
        )

        contextual_top1 = (
            get_score(contextual_results[0])
            if contextual_results
            else 0.0
        )

        # 맥락 검색이 단독 검색보다 좋을 때만 교체한다.
        if (
            contextual_top1 >= LEGAL_MIN_SCORE
            and contextual_top1 > standalone_top1
        ):
            search_results = contextual_results

    final_top1 = (
        get_score(search_results[0])
        if search_results
        else 0.0
    )

    # 모든 검색이 끝난 뒤 최종 점수로 비법률 질문을 차단한다.
    if final_top1 <= HARD_REJECT_SCORE:
        return ChatAnswerResponse(
            answer="죄송합니다. 법률 관련 질문에만 답변할 수 있습니다.",
            verified=True,
            warnings=[],
            unverified_refs=[],
            sources=[],
        )

    if not is_legal_domain(search_results):
        return ChatAnswerResponse(
            answer="죄송합니다. 법률 관련 질문에만 답변할 수 있습니다.",
            verified=True,
            warnings=[],
            unverified_refs=[],
            sources=[],
        )

    # Hybrid에서는 사용자 문서 2개와 공용 법률 3개를 선택한다.
    llm_search_results = select_llm_search_results(
        search_results=search_results,
        context_mode=context_mode,
        document_k=2,
        public_k=3,
    )

    messages = assemble_messages(
        question=question,
        search_results=llm_search_results,
        history=history,
    )

    answer = await generate_answer(messages)

    return build_response(
        answer,
        llm_search_results,
    )