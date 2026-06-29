"""
# RAG 파이프라인 전체(검색→거절판단→프롬프트→LLM→환각검증)를 순서대로 지휘 
# 실제 작업은 다 다른 서비스에 위임
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


HARD_REJECT_SCORE = 1.5
CONTEXT_SEARCH_MAX = 2.0

# 질문을 받고 첫 질문 시 바로 반환 이전 질문이 존재하고 현재 질문과 다르다면 이전질문과 함께 반환
def build_contextual_query(question: str, history: list[dict]) -> str:
    """후속 질문이면 직전 질문 하나를 검색어에 붙인다."""

    if not history:
        return question

    previous_question = (history[0].get("question") or "").strip()

    if previous_question and previous_question != question:
        return f"{previous_question} {question}"

    return question

# result에 score값을 가져와 반환한다 이때 score가 비어있다면 0.0을 반환
def get_score(result: dict) -> float:
    try:
        return float(result.get("score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def select_llm_search_results(
    search_results: list[dict],
    context_mode: str,
    document_k: int = 3,
    public_k: int = 2,
) -> list[dict]:
    """
    파일 업로드 세션에서는:
    - 사용자 문서 3개
    - 공용 법률 2개

    파일이 없는 세션에서는 점수순 상위 5개를 사용한다.
    """

    total_k = document_k + public_k

    # 문서가 hybrid가 아니라면 상위 5개 잘라서 끝
    if context_mode != "hybrid":
        return search_results[:total_k]

    # 개인 문서를 result에 append 하기 
    document_results = [
        result
        for result in search_results
        if result.get("source_type") == "document"
    ]

    # 공용 문서를 result에 append 하기  
    public_results = [
        result
        for result in search_results
        if result.get("source_type") == "legal_vector"
    ]

    # 점수 높은거로 정렬
    document_results.sort(key=get_score, reverse=True)
    public_results.sort(key=get_score, reverse=True)

    # selected에 각 k값에 맞게 넣기
    selected = (
        document_results[:document_k]
        + public_results[:public_k]
    )

    # 한쪽 검색 결과가 부족하면 남은 고득점 결과로 채운다.
    if len(selected) < total_k:
        selected_ids = {
            (result.get("source_type"), result.get("id"))
            for result in selected
        }

        remaining = [
            result
            for result in search_results
            if (result.get("source_type"), result.get("id"))
            not in selected_ids
        ]

        remaining.sort(key=get_score, reverse=True)
        selected.extend(remaining[: total_k - len(selected)])

    selected.sort(key=get_score, reverse=True)
    return selected

# 검색 품질 검증(멀티턴 이탈 등) 필요할 때 주석 해제
"""def print_topk_scores(
    title: str,
    search_results: list[dict],
) -> None:
    print(f"\n========== {title} ==========")

    if not search_results:
        print("검색 결과 없음")
        return

    for index, result in enumerate(search_results, start=1):
        metadata = result.get("metadata") or {}

        content = (
            metadata.get("text")
            or metadata.get("source_text")
            or metadata.get("qa_text")
            or ""
        )

        print(
            f"[{index}] "
            f"score={get_score(result):.4f} | "
            f"source_type={result.get('source_type')} | "
            f"article={metadata.get('article')!r} | "
            f"category={metadata.get('category')!r} | "
            f"text_len={len(str(content))}"
        )"""


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
    """
    질문 하나를 받아 검색부터 답변 생성까지 실행한다.
    """

    # DB가 연결돼 있고, 지금 모드가 기본값인 general이면 실행
    if db is not None and context_mode == "general":
        has_file = (
            # document 테이블에서 session_id가 일치하고 status가 ready or embedded 행 중 
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
            #첫번째 것 가져와
            .first()
            # 없으면 None
            is not None
        )
        # 존재하면 hybrid로 context_mode 변환
        if has_file:
            context_mode = "hybrid"

    # 이전 대화 조회
    if history is None:
        history = get_recent_messages(
            db=db,
            session_id=session_id,
            limit=history_limit,
        )

    # 존재하는 값들을 search_pinecone 함수를 통해 
    # 벡터DB에 검색을 보내고 받아온다
    standalone_results = search_pinecone(
        question=question,
        context_mode=context_mode,
        user_id=user_id,
        top_k=top_k,
        db=db,
    )

    # 받아온 목록 확인
    """print_topk_scores(
        "단독 검색 Top-K",
        standalone_results,
    )"""

    # 가져온 파일의 점수 반환
    standalone_top1 = (
        get_score(standalone_results[0])
        if standalone_results
        else 0.0
    )

    search_results = standalone_results

    # 가져온 파일의 점수를 설정값과 비교하여 걸러내기
    if standalone_top1 <= HARD_REJECT_SCORE:
        return ChatAnswerResponse(
            answer="죄송합니다. 법률 관련 질문에만 답변할 수 있습니다.",
            verified=True,
            warnings=[],
            unverified_refs=[],
            sources=[],
        )

    # 애매한 후속 질문은 직전 질문과 함께 재검색
    # 이전 질문이 없다면 패스
    if (
        standalone_top1 <= CONTEXT_SEARCH_MAX
        and history
    ):
        # 위에서 확인했든 이전 질문이 있다면 같이 들어감
        contextual_query = build_contextual_query(
            question=question,
            history=history,
        )

        # 이전 질문과 함께 들어가 serch_pinecone를 통해 각 값을 검색
        contextual_results = search_pinecone(
            question=contextual_query,
            context_mode=context_mode,
            user_id=user_id,
            top_k=top_k,
            db=db,
        )

        """print_topk_scores(
            "맥락 검색 Top-K",
            contextual_results,
        )"""

        contextual_top1 = (
            get_score(contextual_results[0])
            if contextual_results
            else 0.0
        )
        # 이전 질문과 현재질문의 점수를 뽑아온걸로 비교
        #=================== 이거 테스트 해서 수치 더 올려야 할 수 있음
        if contextual_top1 >= LEGAL_MIN_SCORE:
            search_results = contextual_results

    # 최종 법률 질문 확인
    if not is_legal_domain(search_results):
        return ChatAnswerResponse(
            answer="죄송합니다. 법률 관련 질문에만 답변할 수 있습니다.",
            verified=True,
            warnings=[],
            unverified_refs=[],
            sources=[],
        )

    # 사용자 문서 3개 + 공용 법률 2개 선택
    llm_search_results = select_llm_search_results(
        search_results=search_results,
        context_mode=context_mode,
        document_k=3,
        public_k=2,
    )
    # topk 뽑기
    """print("\n========== 모델 참고자료 ==========")
    print(f"context_mode={context_mode}")"""

    """for index, result in enumerate(
        llm_search_results,
        start=1,
    ):
        print(
            f"[{index}] "
            f"source_type={result.get('source_type')} | "
            f"score={get_score(result):.4f}"
        )"""

    # 프롬프트 조립
    messages = assemble_messages(
        question=question,
        search_results=llm_search_results,
        history=history,
    )

    # 답변 생성
    answer = await generate_answer(messages)

    # 환각 검증 및 출처 생성
    return build_response(
        answer,
        llm_search_results,
    )