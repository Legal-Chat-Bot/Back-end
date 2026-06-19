from app.services.prompts import SYSTEM_PROMPT


# ① 검색 결과 → 참고자료 문자열
def format_search_results(search_results: list[dict]) -> str:
    # 검색 결과가 비어있으면 빈 채로 두지 않고 자료 못 찾음이라고 명시
    if not search_results:
        return "[참고자료]\n관련 자료를 찾을 수 없습니다."

    # 결과 문자열을 한 줄씩 쌓는 리스트 (첫 줄 제목은 [참고자료])
    lines = ["[참고자료]"]
    for i, result in enumerate(search_results, start=1):
        source_type = result["source_type"]
        metadata = result["metadata"]
        # 공용 법령일 때
        if source_type == "legal_vector":
            content = metadata.get("text") or metadata.get("qa_text", "")
        # 개인 문서일 때
        else:  # document
            content = metadata.get("text", "")

        # 관련 조문(article)이 메타데이터에 있으면 본문 앞에 명시
        #   - 형식: "○○법 제○○조" 문자열 하나
        #   - 모델이 조문 번호를 외워서 지어내지 않고, 여기 있는 걸 보고 인용하게 함
        #   - 환각(가짜 조문) 감소 + 검증 정확도 향상
        article = metadata.get("article", "")
        if article:
            lines.append(f"{i}. [관련 조문: {article}] {content}")
        else:
            lines.append(f"{i}. [조항 번호 인용 금지] {content}")   # ← 추가
    return "\n".join(lines)


# ② 조립 → messages 완성
#
# [핵심] 이전 대화(history)를 user 텍스트에 박지 않고,
#        messages 배열의 실제 대화 턴(role: user / assistant)으로 넣는다.
#        → 모델이 흐름으로 인식 → 이전 답변 복사 방지
#        → Ollama /api/chat 의 정석 방식
def assemble_messages(
    question: str,                    # 이번 사용자 질문
    search_results: list[dict],       # search_pinecone가 준 검색 결과
    history: list[dict],              # 이전 대화 [{question, answer}, ...] (최신순)
) -> list[dict]:

    # 1. 시스템 프롬프트
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 2. 이전 대화를 실제 대화 턴으로 추가 (오래된 것부터)
    #    history는 최신순(최근이 앞)으로 들어오므로 reversed로 시간순으로 펼침
    for turn in reversed(history):
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})

    # 3. 이번 질문 = 참고자료 + 질문 (이번 턴의 user 메시지)
    search_str = format_search_results(search_results)
    user_content = f"{search_str}\n\n[질문]\n{question}"
    messages.append({"role": "user", "content": user_content})

    return messages