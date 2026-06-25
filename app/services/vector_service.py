import uuid

from sqlalchemy.orm import Session

import app.db.vector.client as pinecone
from app.db.vector.embedding import embed_query as embed_full
from app.db.models.chunk import Chunk
from app.db.models.chunk_dataset import ChunkDataset


def embed_query(text: str):
    return embed_full(text)


def _to_uuid(value) -> uuid.UUID | None:
    """Pinecone의 문자열 ID를 PostgreSQL UUID로 변환."""
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError, AttributeError):
        return None


def _attach_chunk_text(
    db: Session | None,
    results: list[dict],
) -> list[dict]:
    """
    Pinecone 검색 결과의 vector_id로 RDB 본문을 조회하고
    metadata["text"]에 결합한다.
    """

    # 기존 수동 적재 데이터의 source_text 임시 지원
    for result in results:
        metadata = result["metadata"]

        if not metadata.get("text") and metadata.get("source_text"):
            metadata["text"] = metadata["source_text"]

    if db is None or not results:
        return results

    public_ids: set[uuid.UUID] = set()
    document_ids: set[uuid.UUID] = set()

    # Top-K의 vector_id 수집
    for result in results:
        metadata = result["metadata"]
        raw_id = metadata.get("vector_id") or result.get("id")
        vector_id = _to_uuid(raw_id)

        # medical_law_1처럼 UUID가 아니면 RDB 조회 불가능
        if vector_id is None:
            continue

        if result["source_type"] == "legal_vector":
            public_ids.add(vector_id)
        else:
            document_ids.add(vector_id)

    # 공용 법률 데이터 본문을 한 번에 조회
    public_texts: dict[str, str] = {}

    if public_ids:
        rows = (
            db.query(ChunkDataset)
            .filter(ChunkDataset.vector_id.in_(public_ids))
            .all()
        )

        public_texts = {
            str(row.vector_id): row.chunk_text
            for row in rows
        }

    # 사용자 업로드 문서 본문을 한 번에 조회
    document_texts: dict[str, str] = {}

    if document_ids:
        rows = (
            db.query(Chunk)
            .filter(Chunk.vector_id.in_(document_ids))
            .all()
        )

        document_texts = {
            str(row.vector_id): row.chunk_text
            for row in rows
        }

    # Pinecone metadata에 RDB 본문 결합
    for result in results:
        metadata = result["metadata"]
        raw_id = metadata.get("vector_id") or result.get("id")
        vector_id = _to_uuid(raw_id)

        if vector_id is None:
            continue

        if result["source_type"] == "legal_vector":
            chunk_text = public_texts.get(str(vector_id))
        else:
            chunk_text = document_texts.get(str(vector_id))

        if chunk_text:
            metadata["text"] = chunk_text

    return results


def search_pinecone(
    question: str,
    context_mode: str,
    user_id: str,
    top_k: int = 5,
    db: Session | None = None,
) -> list[dict]:
    emb = embed_full(question)

    def to_result(match: dict, source_type: str) -> dict:
        return {
            "id": match.get("id", ""),
            "score": match["score"],
            "source_type": source_type,
            "metadata": dict(match.get("metadata", {})),
        }

    if context_mode == "general":
        matches = pinecone.query(
            dense_vector=emb.dense,
            sparse_vector=emb.sparse,
            namespace=pinecone.public_namespace(),
            top_k=top_k,
        )

        results = [
            to_result(match, "legal_vector")
            for match in matches
        ]
        results.sort(key=lambda item: item["score"], reverse=True)

        return _attach_chunk_text(db, results[:top_k])

    if context_mode == "document":
        matches = pinecone.query(
            dense_vector=emb.dense,
            sparse_vector=emb.sparse,
            namespace=pinecone.user_namespace(user_id),
            top_k=top_k,
        )

        results = [
            to_result(match, "document")
            for match in matches
        ]
        results.sort(key=lambda item: item["score"], reverse=True)

        return _attach_chunk_text(db, results[:top_k])

    if context_mode == "hybrid":
        document_k = max(1, round(top_k * 0.6))
        public_k = top_k - document_k

        private_matches = pinecone.query(
            dense_vector=emb.dense,
            sparse_vector=emb.sparse,
            namespace=pinecone.user_namespace(user_id),
            top_k=top_k,
        )

        public_matches = pinecone.query(
            dense_vector=emb.dense,
            sparse_vector=emb.sparse,
            namespace=pinecone.public_namespace(),
            top_k=top_k,
        )

        private_results = [
            to_result(match, "document")
            for match in private_matches
        ]
        public_results = [
            to_result(match, "legal_vector")
            for match in public_matches
        ]

        private_results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )
        public_results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        results = (
            private_results[:document_k]
            + public_results[:public_k]
        )

        # 결과가 부족하면 남은 후보로 채움
        if len(results) < top_k:
            remaining = (
                private_results[document_k:]
                + public_results[public_k:]
            )
            remaining.sort(
                key=lambda item: item["score"],
                reverse=True,
            )
            results.extend(remaining[:top_k - len(results)])

        results.sort(key=lambda item: item["score"], reverse=True)

        return _attach_chunk_text(db, results[:top_k])

    raise ValueError(f"Invalid context_mode: {context_mode}")


def is_legal_domain(
    search_results: list[dict],
    threshold: float = 0.6,
) -> bool:
    if not search_results:
        return False

    return search_results[0]["score"] >= threshold


def is_legal_file(
    extracted_text: str,
    threshold: float = 0.6,
) -> bool:
    results = search_pinecone(
        question=extracted_text[:500],
        context_mode="general",
        user_id="",
        top_k=3,
    )

    return is_legal_domain(results, threshold)