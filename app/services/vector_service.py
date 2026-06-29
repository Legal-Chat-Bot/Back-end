import re
import uuid

from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

import app.db.vector.client as pinecone_client
from app.db.models.chunk import Chunk
from app.db.vector.embedding import embed_query as embed_full


LEGAL_MIN_SCORE = 0.99
LAW_NAME_PATTERN = re.compile(r"([가-힣A-Za-z0-9·ㆍ]+법)")


def _to_uuid(value) -> uuid.UUID | None:
    """문자열을 UUID로 변환한다."""

    if not value:
        return None

    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError):
        return None


def _get_score(result: dict) -> float:
    """Pinecone 검색 점수를 안전하게 반환한다."""

    try:
        return float(result.get("score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _get_embedding_value(embedding, *names):
    """dict 또는 객체 형태의 임베딩 결과에서 값을 꺼낸다."""

    if isinstance(embedding, dict):
        for name in names:
            value = embedding.get(name)
            if value is not None:
                return value

    for name in names:
        value = getattr(embedding, name, None)
        if value is not None:
            return value

    return None

# 공용과 사용자 문서를 구분한다
def _get_source_type(namespace: str, result: dict) -> str:

    metadata = result.get("metadata") or {}

    # 
    if metadata.get("source_type"):
        return metadata["source_type"]

    if namespace == "public":
        return "legal_vector"

    return "document"

# 벡터DB 검색결과를 통일된 형태로 통일 시킴
def _normalize_result(result: dict, namespace: str) -> dict:

    metadata = dict(result.get("metadata") or {})
    vector_id = metadata.get("vector_id") or result.get("id")

    if vector_id and not metadata.get("vector_id"):
        metadata["vector_id"] = vector_id

    return {
        "id": result.get("id"),
        "score": _get_score(result),
        "source_type": _get_source_type(namespace, result),
        "metadata": metadata,
    }

# 벡터ID를 사용하여 RDB에서 본문 text를 가져온다 
def _attach_chunk_text(
    db: Session | None,
    results: list[dict],
) -> list[dict]:

    # Pinecone metadata에 source_text가 있으면 text로 사용
    for result in results:
        metadata = result.get("metadata") or {}

        if not metadata.get("text") and metadata.get("source_text"):
            metadata["text"] = metadata["source_text"]

        result["metadata"] = metadata

    if db is None or not results:
        return results

    public_ids: set[uuid.UUID] = set()
    document_ids: set[uuid.UUID] = set()

    # 현재 Top-K 결과에 포함된 ID만 수집
    # 각 검색 결과에서 vector_id를 뽑고 공용과 개인에 나눠 벡터id 저장
    for result in results:
        metadata = result.get("metadata") or {}
        raw_id = metadata.get("vector_id") or result.get("id")
        vector_id = _to_uuid(raw_id)

        if vector_id is None:
            continue

        if result.get("source_type") == "legal_vector":
            public_ids.add(vector_id)
        else:
            document_ids.add(vector_id)

    public_texts: dict[str, str] = {}
    public_articles: dict[str, str] = {}

    # 위에 보관한 id를 통해 본문과 조문을 RDB에서 찾아오는 (공용)
    if public_ids:
        statement = text(
            """
            SELECT vector_id, chunk_text, article
            FROM chunk_dataset
            WHERE vector_id::text IN :vector_ids
            """
        ).bindparams(bindparam("vector_ids", expanding=True))

        rows = db.execute(
            statement,
            {"vector_ids": [str(value) for value in public_ids]},
        ).mappings().all()

        public_texts = {
            str(row["vector_id"]): row["chunk_text"]
            for row in rows
            if row.get("chunk_text")
        }

        public_articles = {
            str(row["vector_id"]): row["article"]
            for row in rows
            if row.get("article")
        }

    document_texts: dict[str, str] = {}
    document_articles: dict[str, str] = {}

    # 위에 보관한 id를 통해 본문과 조문을 RDB에서 찾아오는 (개인)
    if document_ids:
        rows = (
            db.query(Chunk)
            .filter(Chunk.vector_id.in_(document_ids))
            .all()
        )

        document_texts = {
            str(row.vector_id): row.chunk_text
            for row in rows
            if getattr(row, "chunk_text", None)
        }

        document_articles = {
            str(row.vector_id): row.article
            for row in rows
            if getattr(row, "article", None)
        }

    # 검색 결과에 RDB 본문과 조문 결합
    for result in results:
        metadata = result.get("metadata") or {}
        raw_id = metadata.get("vector_id") or result.get("id")
        vector_id = _to_uuid(raw_id)

        if vector_id is None:
            continue

        key = str(vector_id)

        if result.get("source_type") == "legal_vector":
            chunk_text = public_texts.get(key)
            article = public_articles.get(key)
        else:
            chunk_text = document_texts.get(key)
            article = document_articles.get(key)

        if chunk_text and not metadata.get("text"):
            metadata["text"] = chunk_text

        if article and not metadata.get("article"):
            metadata["article"] = article

        result["metadata"] = metadata

    return results


def search_pinecone(
    question: str,
    context_mode: str = "general",
    user_id: str | None = None,
    top_k: int = 5,
    db: Session | None = None,
) -> list[dict]:
    """
    질문을 임베딩하고 Pinecone에서 검색한다.

    general: 공용 법률
    document: 사용자 문서
    hybrid: 공용 법률 + 사용자 문서
    """

    embedding = embed_full(question)

    dense_vector = _get_embedding_value(
        embedding,
        "dense_vecs",
        "dense",
        "dense_vector",
    )

    sparse_vector = _get_embedding_value(
        embedding,
        "lexical_weights",
        "sparse",
        "sparse_vector",
    )

    if dense_vector is None:
        raise ValueError(
            "임베딩 결과에서 dense vector를 찾을 수 없습니다."
        )

    results: list[dict] = []

    # 공용 법률 검색
    if context_mode in ("general", "hybrid"):
        public_matches = pinecone_client.query(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            namespace="public",
            top_k=top_k,
            filter=None,
        )

        for match in public_matches:
            results.append(
                _normalize_result(match, namespace="public")
            )

    # 사용자 업로드 문서 검색
    if context_mode in ("document", "hybrid") and user_id:
        user_namespace = str(user_id)

        user_matches = pinecone_client.query(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            namespace=user_namespace,
            top_k=top_k,
            filter=None,
        )

        for match in user_matches:
            results.append(
                _normalize_result(
                    match,
                    namespace=user_namespace,
                )
            )

    # 점수순 정렬 후 최종 Top-K 선택
    results.sort(key=_get_score, reverse=True)
    results = results[:top_k]

    # RDB 본문 결합
    return _attach_chunk_text(
        db=db,
        results=results,
    )

# 검색한 결과 top1이 지정한 수치를 넘어서 
# 이 검색이 법률 도메인에 충분히 가까운가/품질이 괜찮은가를 판단
"""def evaluate_search_quality(
    question: str,
    search_results: list[dict],
    min_top1_score: float = LEGAL_MIN_SCORE,
) -> tuple[bool, str]:

    if not search_results:
        return False, "검색 결과 없음"

    top1_score = _get_score(search_results[0])

    if top1_score < min_top1_score:
        return (
            False,
            f"Top-1 점수 미달: "
            f"{top1_score:.4f} < {min_top1_score:.4f}",
        )

    return (
        True,
        f"Top-1 점수 통과: {top1_score:.4f}",
    )"""


def is_legal_domain(
    search_results: list[dict],
    min_top1_score: float = LEGAL_MIN_SCORE,
) -> bool:
    """
    Top-1 검색 점수를 기준으로 법률 질문 여부를 판단한다.

    법률 전용 인덱스에서는 비법률 질문도 legal_vector 결과를
    반환하므로 source_type이나 metadata 존재 여부로 판단하지 않는다.
    """

    if not search_results:
        return False

    top1_score = _get_score(search_results[0])
    return top1_score >= min_top1_score