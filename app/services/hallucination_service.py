def _collect_topk_refs(search_results: list[dict]) -> set[str]:
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