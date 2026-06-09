from app.services.prompts import SYSTEM_PROMPT

# ① 검색 결과 → 참고자료 문자열
def format_search_results(search_results: list[dict]) -> str:
    if not search_results:
        return "[참고자료]\n관련 자료를 찾을 수 없습니다."

    lines = ["[참고자료]"]
    for i, result in enumerate(search_results, start=1):
        source_type = result["source_type"]
        metadata = result["metadata"]

        if source_type == "legal_vector":
            source_label = metadata.get("doc_title") or metadata.get("doc_source", "법령/판례")
            content = metadata.get("text") or metadata.get("qa_text", "")
        else:  # document
            source_label = metadata.get("filename", "업로드 문서")
            content = metadata.get("text", "")

        lines.append(f"{i}. (출처: {source_label}) {content}")

    return "\n".join(lines)

# ② 대화이력 → 이전 대화 문자열
def format_history(history: list[dict]) -> str:
    if not history:
        return ""

    lines = ["[이전 대화]"]
    for turn in reversed(history):   # 오래된 것부터 보여줘야 자연스러운 흐름
        lines.append(f"사용자: {turn['question']}")
        lines.append(f"어시스턴트: {turn['answer']}")

    return "\n".join(lines)


# ③ 조립 → messages 완성
def assemble_messages(
    question: str,
    search_results: list[dict],
    history: list[dict],
) -> list[dict]:
    context_parts = []

    context_parts.append(format_search_results(search_results))

    history_str = format_history(history)
    if history_str:
        context_parts.append(history_str)

    context_parts.append(f"[질문]\n{question}")

    user_content = "\n\n".join(context_parts)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]