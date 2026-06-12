from app.services.prompts import SYSTEM_PROMPT

# 입력 : search_results -> search_pinecone이 준 검색 결과 리스트
def format_search_results(search_results: list[dict]) -> str:
    # 검색 결과가 비어있으면 빈채로 두지않고 자료 못찾음 이라고 명시
    if not search_results:
        return "[참고자료]\n관련 자료를 찾을 수 없습니다."
    
    # 결과 문제열을 한 줄씩 쌓는 리스트 첫 줄 제목은 [참고자료]
    # 나중에 이 리스트를 합쳐서 하나의 문열로 만듬
    lines = ["[참고자료]"]
    # enumerate -> 리스트를 돌면서 번호(i)랑 항목(result)을 같이 줌 
    for i, result in enumerate(search_results, start=1):
        # 각 결과에서 출처 종류와 메타데이터를 꺼냄
        source_type = result["source_type"]
        metadata = result["metadata"]
        # 공용 범령일때
        if source_type == "legal_vector":
            # 출처 이름을 문서 제목 먼저 뽑고
            # 만약 제목이 없으면 출처 
            # 그래도 없으면 범령/판례로 뽑음
            source_label = metadata.get("doc_title") or metadata.get("doc_source", "법령/판례")
            # 실제 문서 본문
            # 없으면 Q&A 텍스트
            # 그것도 없으면 빈문자열
            content = metadata.get("text") or metadata.get("qa_text", "")
        # 개인문서일때
        else:  # document
            # 파일명 없으면 업로드 문서
            source_label = metadata.get("filename", "업로드 문서")
            # 본문 없으면 공백
            content = metadata.get("text", "")
        # 번호, 출처, 내용을 한줄로 만들어 lines에 추가
        lines.append(f"{i}. (출처: {source_label}) {content}")
    # 전부 합쳐서 이어붙여 반환
    return "\n".join(lines)

# ② 대화이력 → 이전 대화 문자열
def format_history(history: list[dict]) -> str:
    # 이전 대화가 없으면(첫 질문이면) 빈 문자열 반환
    if not history:
        return ""
    
    # 줄들을 쌓을 리스트 첫 줄 제목은 [이전대화]
    lines = ["[이전 대화]"]
    # 오래된 것부터 읽어야 자연스러워 reversed 해줌
    for turn in reversed(history):   # 오래된 것부터 보여줘야 자연스러운 흐름
        # turn에서 질문과 답변을 꺼내서 붙임
        lines.append(f"사용자: {turn['question']}")
        lines.append(f"어시스턴트: {turn['answer']}")
    
    # 줄들을 하나의 문자열로 완성
    return "\n".join(lines)


# ③ 조립 → messages 완성
def assemble_messages(
    # 사용자 질문
    question: str,
    # 검색결과(search_pinecone가 준거)
    search_results: list[dict],
    # 이전 대화 기록
    history: list[dict],
) -> list[dict]:
    
    context_parts = []
    # search_results로 검색 결과를 텍스트로 만들고 붙인다
    context_parts.append(format_search_results(search_results))
    
    # 이전 대화 추가(있을 때만)
    history_str = format_history(history)
    # 결과가 비어있지 않을 때
    if history_str:
        context_parts.append(history_str)

    context_parts.append(f"[질문]\n{question}")

    user_content = "\n\n".join(context_parts)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]