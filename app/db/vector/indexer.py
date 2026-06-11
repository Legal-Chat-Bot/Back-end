

# 인덱싱 파이프라인
# 문서 → 거름망 → 청킹 → DB 저장(chunk_id 확보) → 임베딩 → Pinecone upsert

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

import app.db.vector.client as pinecone
from app.db.vector.embedding import embed_texts
from app.db.vector.chunker import Chunker,Chunk


@dataclass
class IndexingResult:
    document_id: str
    doc_type: str
    total_chunks: int
    namespace: str


# ✅ 청킹 초기화 - ChunkConfig 기본값 사용 (chunker.py에서 관리)
_chunker = Chunker()


# ── 텍스트 거름망 ─────────────────────────────────────────
def _filter_text(text: str) -> str | None:
    """너무 짧거나 의미 없는 텍스트 걸러냄"""
    text = text.strip()
    if len(text) < 20:
        return None
    return text


# ── Pinecone 벡터 조립 ────────────────────────────────────
def build_pinecone_vectors(
    chunks: list[Chunk],
    chunk_ids: list[str],
    category: str,
    law_date: str,
    embeddings: list[list[float]],
    sparse_vectors: list[dict],
) -> list[dict]:
    """청크 + 임베딩 → Pinecone upsert용 벡터 리스트 조립"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    vectors = []

    for chunk, chunk_id, dense, sparse in zip(chunks, chunk_ids, embeddings, sparse_vectors):
        vectors.append({
            "id": str(uuid4()),
            "values": dense,
            "sparse_values": sparse,
            "metadata": {
                "vector_id":   chunk_id,        # RDB chunks 테이블 id 참조용
                "chunk_index": chunk.chunk_index,  # ✅ 추가
                "text":        chunk.text,         # ✅ 추가 (검색 결과에서 텍스트 바로 확인용)
                "category":    category,
                "created_at":  now,
                "updated_at":  now,
                "law_date":    law_date,
            },
        })
    return vectors


# ── 법률 문서 인덱싱 ──────────────────────────────────────
async def index_public_document(
    text: str,
    document_id: UUID,
    category: str,
    law_date: str,
) -> IndexingResult:
    namespace = pinecone.public_namespace()

    # 1. 거름망
    filtered = _filter_text(text)
    if filtered is None:
        raise ValueError(f"[{document_id}] 텍스트가 너무 짧거나 비어있음")

    # 2. 청킹
    chunks = _chunker.chunk(filtered, already_cleaned=True)
    if not chunks:
        raise ValueError(f"[{document_id}] 청킹 결과 없음")

    # 3. 임베딩
    texts = [c.text for c in chunks]
    embedding_results = embed_texts(texts)

    # ✅ 임베딩 수 검증
    if len(embedding_results) != len(chunks):
        raise ValueError(
            f"[{document_id}] 임베딩 수 불일치: 청크={len(chunks)}, 임베딩={len(embedding_results)}"
        )

    # 4. Pinecone 벡터 조립
    chunk_ids = [str(uuid4()) for _ in chunks]
    vectors = build_pinecone_vectors(
        chunks=chunks,
        chunk_ids=chunk_ids,
        category=category,
        law_date=law_date,
        embeddings=[e.dense for e in embedding_results],
        sparse_vectors=[e.sparse for e in embedding_results],
    )

    # 5. Pinecone upsert
    pinecone.upsert(vectors=vectors, namespace=namespace)

    return IndexingResult(
        document_id=str(document_id),
        doc_type=category,
        total_chunks=len(chunks),
        namespace=namespace,
    )


# ── 유저 문서 인덱싱 ──────────────────────────────────────
async def index_user_document(
    text: str,
    document_id: UUID,
    user_id: str,
    category: str,
) -> IndexingResult:
    namespace = pinecone.user_namespace(user_id)

    # 1. 거름망
    filtered = _filter_text(text)
    if filtered is None:
        raise ValueError(f"[{document_id}] 텍스트가 너무 짧거나 비어있음")

    # 2. 청킹
    chunks = _chunker.chunk(filtered)
    if not chunks:
        raise ValueError(f"[{document_id}] 청킹 결과 없음")

    # 3. 임베딩
    texts = [c.text for c in chunks]
    embedding_results = embed_texts(texts)

    # ✅ 임베딩 수 검증
    if len(embedding_results) != len(chunks):
        raise ValueError(
            f"[{document_id}] 임베딩 수 불일치: 청크={len(chunks)}, 임베딩={len(embedding_results)}"
        )

    # 4. Pinecone 벡터 조립
    chunk_ids = [str(uuid4()) for _ in chunks]
    vectors = build_pinecone_vectors(
        chunks=chunks,
        chunk_ids=chunk_ids,
        category=category,
        law_date="",   # 유저 문서는 law_date 없음
        embeddings=[e.dense for e in embedding_results],
        sparse_vectors=[e.sparse for e in embedding_results],
    )

    # 5. Pinecone upsert
    pinecone.upsert(vectors=vectors, namespace=namespace)

    return IndexingResult(
        document_id=str(document_id),
        doc_type=category,
        total_chunks=len(chunks),
        namespace=namespace,
    )
