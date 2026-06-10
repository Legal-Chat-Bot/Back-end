# app/services/vector_service.py

from FlagEmbedding import BGEM3FlagModel
from pinecone import Pinecone
from app.core.config import settings

# ── 모델/클라이언트는 모듈 로드 시 1회만 초기화 ──
# 매 요청마다 모델 로드하면 느리니까 전역으로 한 번만 올림
_embed_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
_pc = Pinecone(api_key=settings.PINECONE_API_KEY)
_index = _pc.Index(settings.PINECONE_INDEX_NAME)


# ① 질문 임베딩
def embed_query(text: str) -> list[float]:
    """질문 텍스트 → 1024차원 벡터"""
    result = _embed_model.encode([text])
    # BGE-M3는 dense_vecs에 dense 임베딩이 들어있음
    return result["dense_vecs"][0].tolist()


# ② 검색 (context_mode 분기 + user_id 필터 + 병합)
def search_pinecone(
    query_vector: list[float],
    context_mode: str,
    user_id: str,
    top_k: int = 5,
) -> list[dict]:
    """
    context_mode에 따라 공용/개인 Pinecone 검색 후 병합
    - general  : 공용만
    - document : 개인만 (user_id 필터)
    - hybrid   : 둘 다 검색 후 병합
    """
    results = []

    # 공용 DB 검색 (general, hybrid)
    if context_mode in ("general", "hybrid"):
        public_res = _index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            # 공용 네임스페이스 (개인과 분리 저장한다는 전제)
            #namespace="public",
        )
        for m in public_res["matches"]:
            results.append({
                "score": m["score"],
                "source_type": "legal_vector",   # 문서 enum: legal_vector
                "metadata": m["metadata"],
            })

    # 개인 DB 검색 (document, hybrid) — user_id 필터 필수!
    if context_mode in ("document", "hybrid"):
        private_res = _index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            namespace="private",
            filter={"user_id": user_id},   # ← 남의 문서 차단
        )
        for m in private_res["matches"]:
            results.append({
                "score": m["score"],
                "source_type": "document",   # 문서 enum: document
                "metadata": m["metadata"],
            })

    # 병합 결과를 유사도 높은 순으로 정렬 후 top_k개만
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

def is_legal_file(extracted_text, threshold = 0.6):
    """
    파일에서 추출한 텍스트가 법률 도메인인지 판별
    질문 거름망이랑 동일한 로직 / 입력만 다름
    """
    vector = embed_query(extracted_text[:500])
    results = search_pinecone(
        query_vector=vector,
        context_mode='general',
        user_id="",
        top_k=3,
    )
    return is_legal_domain(results, threshold)