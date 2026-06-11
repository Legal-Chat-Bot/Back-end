"""
6단계: 환각 검증 + 출처 포맷

역할:
1. LLM 답변에서 법령 조문(예: 형법 제268조)을 추출
2. 추출한 조문이 Top-K의 메타데이터(article)에 실제로 있는지 대조
3. 없으면 "참고 자료에서 확인되지 않음" 경고 생성 (답변 자체는 건드리지 않음)
4. 사용자에게 보여줄 출처(sources) 포맷 생성

검증 단위: '조'. 항은 검증하지 않음 (메타데이터에 조만 넣기로 약속)
대조 대상: 메타데이터 article 우선, 없으면 본문(text) 폴백
"""

import re

# ── 법령 조문 추출용 정규식 ──
# "형법 제268조", "관세법 제30조", "관세법 시행령 제17조", "민법 제839조의2" 등
# 법 이름(한글+법) + (시행령 등 선택) + 제N조 + (의N 선택) 까지 한 덩어리로 추출
# 법 이름이 붙은 것만 잡음 → 어느 법인지 알 수 있어야 정확히 대조 가능
RE_LAW_REF = re.compile(r"[가-힣]+법(?:\s*시행령|\s*시행규칙)?\s*제\d+조(?:의\d+)?")


def _normalize(ref: str) -> str:
    """
    조문 표기 정규화 — 공백 차이로 인한 오탐 방지
    "관세법 제30조" 와 "관세법  제30조" (공백 개수 차이) 를 같게 만듦
    "관세법 시행령 제17조" → "관세법시행령제17조"
    """
    return re.sub(r"\s+", "", ref)


def extract_law_refs(text: str) -> list[str]:
    """
    텍스트(주로 LLM 답변)에서 법령 조문을 추출
    반환: 중복 제거된 조문 리스트 (예: ["형법 제268조", "민법 제839조"])
    """
    refs = RE_LAW_REF.findall(text)
    # 순서 유지하면서 중복 제거
    seen = set()
    unique = []
    for r in refs:
        key = _normalize(r)
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def _collect_topk_refs(search_results: list[dict]) -> set[str]:
    """
    Top-K 검색 결과에서 '정답 조문 집합'을 모음 (정규화된 형태로)

    우선순위:
      1순위: metadata["article"] ("형법 제268조" 형태 문자열로 저장)
      2순위: 없으면 metadata["text"] 본문에서 정규식으로 추출 (폴백)
    """
    valid_refs = set()

    for r in search_results:
        metadata = r.get("metadata", {})

        # 1순위: article 필드 ("형법 제268조" 형태 문자열로 저장)
        article = metadata.get("article")
        if article:  # 빈 문자열("")이면 건너뜀
            valid_refs.add(_normalize(article))
            continue  # article 있으면 본문 폴백 안 함

        # 2순위 (폴백): 본문에서 추출
        text = metadata.get("text") or metadata.get("qa_text", "")
        for ref in RE_LAW_REF.findall(text):
            valid_refs.add(_normalize(ref))

    return valid_refs


def verify_answer(answer: str, search_results: list[dict]) -> dict:
    """
    LLM 답변의 조문이 Top-K에 실제로 있는지 검증 (A방식: 표시만, 답변 미수정)

    반환:
    {
        "verified": bool,              # 환각 없으면 True
        "answer_refs": [...],          # 답변에서 추출된 조문 전체
        "verified_refs": [...],        # Top-K에서 확인된 조문
        "unverified_refs": [...],      # Top-K에 없는 조문 (환각 의심)
        "warnings": [...],             # 사용자용 경고 문구
    }
    """
    # 답변에서 조문 추출
    answer_refs = extract_law_refs(answer)

    # 답변에 조문이 아예 없으면 → 검증할 것 없음 → 통과
    if not answer_refs:
        return {
            "verified": True,
            "answer_refs": [],
            "verified_refs": [],
            "unverified_refs": [],
            "warnings": [],
        }

    # Top-K의 정답 조문 집합
    valid_refs = _collect_topk_refs(search_results)

    verified = []
    unverified = []
    for ref in answer_refs:
        if _normalize(ref) in valid_refs:
            verified.append(ref)
        else:
            unverified.append(ref)

    # 경고 문구 생성 (조항 콕 집기)
    warnings = [
        f"'{ref}'는 참고 자료에서 직접 확인되지 않았습니다. 정확하지 않을 수 있습니다."
        for ref in unverified
    ]

    return {
        "verified": len(unverified) == 0,
        "answer_refs": answer_refs,
        "verified_refs": verified,
        "unverified_refs": unverified,
        "warnings": warnings,
    }


def format_sources(search_results: list[dict]) -> list[dict]:
    """
    Top-K 검색 결과 → 사용자에게 보여줄 출처 리스트

    실제 판례 데이터 구조 기준 (category, text, law_date, article)
    제목 필드가 없으므로 category + 본문 발췌로 표시
    """
    sources = []
    for r in search_results:
        metadata = r.get("metadata", {})
        source_type = r.get("source_type", "")

        if source_type == "document":
            # 개인 업로드 문서
            title = metadata.get("filename", "업로드 문서")
        else:
            # 공용 법령/판례 — 제목 없으면 category로
            title = metadata.get("doc_title") or metadata.get("category", "법령/판례")

        text = metadata.get("text") or metadata.get("qa_text", "")
        excerpt = text[:120].replace("\n", " ").strip()
        if len(text) > 120:
            excerpt += "..."

        sources.append({
            "title": title,
            "category": metadata.get("category"),
            "article": metadata.get("article"),
            "law_date": metadata.get("law_date"),
            "excerpt": excerpt,
            "score": round(r.get("score", 0), 4),
        })

    return sources


def build_response(answer: str, search_results: list[dict]) -> dict:
    """
    최종 응답 조립 — chat_service에서 이거 하나만 부르면 됨

    답변 + 검증결과 + 출처를 한 번에 묶어서 반환
    """
    verification = verify_answer(answer, search_results)
    sources = format_sources(search_results)

    return {
        "answer": answer,                              # 원본 답변 (A방식: 수정 안 함)
        "verified": verification["verified"],          # 환각 없으면 True
        "warnings": verification["warnings"],          # 사용자용 경고 (없으면 빈 리스트)
        "unverified_refs": verification["unverified_refs"],  # 환각 의심 조문
        "sources": sources,                            # 출처 리스트
    }