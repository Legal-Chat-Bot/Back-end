# BGE-M3 모델을 사용하기 위한 클래스
from FlagEmbedding import BGEM3FlagModel
# 클라우드 벡터 데이터 베이스인 Pinecone의 공식 파이썬 라이브러리
from pinecone import Pinecone
# 내가 설정한 객체
from app.core.config import settings

# ── 모델/클라이언트는 모듈 로드 시 1회만 초기화 ──
# 매 요청마다 모델 로드하면 느리니까 전역으로 한 번만 올림

# BAAI/bge-m3 : 허깅페이스에서 받아올 모델 이름
# use_fp16=True : 숫자를 16비트로 저장
_embed_model = BGEM3FlagModel(settings.EMBEDDING_MODEL, use_fp16=True)
# Pincone 서버랑 연결하는 클라이언트 생성, API키로 인증
_pc = Pinecone(api_key=settings.PINECONE_API_KEY)
# 검색, 조회 부분은 _index한테 시킨다
_index = _pc.Index(settings.PINECONE_INDEX_NAME)


# 질문을 숫자 벡터화시키는 파트
def embed_query(text: str) -> list[float]:
    """질문 텍스트 → 1024차원 벡터"""
    # text를 리스트에 감싸 
    result = _embed_model.encode([text])
    # BGE-M3는 dense_vecs에 dense 임베딩이 들어있음
    return result["dense_vecs"][0].tolist()


# 질문 벡터로 Pinecone에서 비슷한 자료 찾아오기
def search_pinecone(
    query_vector: list[float],
    context_mode: str,
    user_id: str,
    top_k: int = 5,
) -> list[dict]:
    results = []

    # 공용 DB 검색 (general, hybrid)
    if context_mode in ("general", "hybrid"):
        public_res = _index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            namespace="public",          # ← 수정 ①
        )
        for m in public_res["matches"]:
            results.append({
                "score": m["score"],
                "source_type": "legal_vector",
                "metadata": m["metadata"],
            })

    # 개인 DB 검색 (document, hybrid)
    if context_mode in ("document", "hybrid"):
        private_res = _index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            namespace=user_id,           # ← 수정 ② (filter 삭제)
        )
        for m in private_res["matches"]:
            results.append({
                "score": m["score"],
                "source_type": "document",
                "metadata": m["metadata"],
            })

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