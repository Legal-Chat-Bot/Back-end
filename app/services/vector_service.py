"""
vector_service: 질문 임베딩 + 하이브리드 검색 + 거름망

- 임베딩: embedding.embed_query (dense + sparse 둘 다 생성)
- 검색:   client.query (dense 70% + sparse 30% 하이브리드, alpha=0.7)
- namespace: client.public_namespace() / client.user_namespace()

기존에 직접 모델/Pinecone을 올리던 코드는 모듈 호출로 대체함.
"""

# 벡터 모듈
import app.db.vector.client as pinecone
# embed_query (dense+sparse 반환) — 이름 충돌 피하려고 embed_full로 가져옴
from app.db.vector.embedding import embed_query as embed_full


# 질문을 임베딩 (dense + sparse)
def embed_query(text: str):
    """
    질문 텍스트 → EmbeddingResult(dense, sparse, text)
    embed_query를 그대로 사용 (하이브리드용 sparse까지 포함)
    """
    return embed_full(text)


# 질문으로 Pinecone 하이브리드 검색
def search_pinecone(
    question: str,
    context_mode: str,     # general / document / hybrid
    user_id: str,
    top_k: int = 5,
) -> list[dict]:
    """
    질문을 받아 하이브리드(dense+sparse) 검색 수행

    변경점:
    - 기존엔 미리 만든 dense 벡터를 받았지만,
      이제 question을 직접 받아 내부에서 embed_full로 dense+sparse 생성
    - 검색은 client.query 사용 (하이브리드 + namespace 함수)
    """
    # 질문 임베딩 (dense + sparse)
    emb = embed_full(question)

    results = []

    # 공용 DB 검색 (general, hybrid) → namespace="public"
    if context_mode in ("general", "hybrid"):
        public_matches = pinecone.query(
            dense_vector=emb.dense,
            sparse_vector=emb.sparse,
            namespace=pinecone.public_namespace(),
            top_k=top_k,
        )
        for m in public_matches:
            results.append({
                "score": m["score"],
                "source_type": "legal_vector",
                "metadata": m["metadata"],
            })

    # 개인 DB 검색 (document, hybrid) → namespace="user_{user_id}"
    if context_mode in ("document", "hybrid"):
        private_matches = pinecone.query(
            dense_vector=emb.dense,
            sparse_vector=emb.sparse,
            namespace=pinecone.user_namespace(user_id),
            top_k=top_k,
        )
        for m in private_matches:
            results.append({
                "score": m["score"],
                "source_type": "document",
                "metadata": m["metadata"],
            })

    # hybrid일 때 공용+개인 결과가 섞이므로 점수순 정렬 후 상위 top_k만
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# ③ 거름망 — 법률 도메인 질문인지 판별
def is_legal_domain(search_results: list[dict], threshold: float = 0.6) -> bool:
    """
    검색 결과 최고 유사도가 임계값 미만이면 법률 질문 아님
    threshold는 나중에 법률/비법률 질문으로 테스트해서 조정
    """
    if not search_results:
        return False
    top_score = search_results[0]["score"]   # 이미 정렬돼 있음
    return top_score >= threshold


# 파일 거름망 — 질문 거름망과 동일 로직 / 입력만 다름
def is_legal_file(extracted_text: str, threshold: float = 0.6) -> bool:
    """
    파일에서 추출한 텍스트가 법률 도메인인지 판별
    전체 임베딩은 느리므로 앞 500자만 잘라서 확인
    """
    results = search_pinecone(
        question=extracted_text[:500],
        context_mode="general",
        user_id="",
        top_k=3,
    )
    return is_legal_domain(results, threshold)