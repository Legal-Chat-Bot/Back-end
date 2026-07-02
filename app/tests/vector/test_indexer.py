"""
단위 테스트: app/db/vector/indexer.py
- embed_texts / pinecone / create_chunks_bulk / _chunker 전부 mock
- 외부 I/O 없이 순수 로직만 검증
"""

import pytest
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace
from dataclasses import dataclass, field
from typing import Optional


# ── 인라인 Stub (실제 프로젝트에서는 from app.db.vector.indexer import ... 로 교체) ──

UPSERT_BATCH_SIZE = 200

@dataclass
class Chunk:
    text: str
    chunk_index: int
    article: str = ""
    law_name: str = ""

@dataclass
class IndexingResult:
    document_id: str
    doc_type: str
    total_chunks: int
    namespace: str
    synced_public_chunks: int = 0


def _filter_text(text: str) -> str | None:
    text = text.strip()
    return None if len(text) < 20 else text


def build_pinecone_vectors(chunks, chunk_ids, document_id, category, law_name,
                           law_date, embeddings, sparse_vectors):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    vectors = []
    for chunk, chunk_id, dense, sparse in zip(chunks, chunk_ids, embeddings, sparse_vectors):
        resolved = law_name or chunk.law_name or f"{category}법"
        article = f"{resolved} {chunk.article}".strip() if chunk.article else resolved
        vectors.append({
            "id": chunk_id,
            "values": dense,
            "sparse_values": sparse,
            "metadata": {
                "vector_id":   chunk_id,
                "document_id": document_id,
                "chunk_index": chunk.chunk_index,
                "article":     article,
                "category":    category,
                "law_date":    law_date,
                "created_at":  now,
                "updated_at":  now,
            },
        })
    return vectors


def _build_article_key(chunk: Chunk, law_name: str, category: str) -> str:
    resolved = law_name or chunk.law_name or f"{category}법"
    return f"{resolved} {chunk.article}".strip() if chunk.article else resolved


def _upsert_in_batches(upsert_fn, vectors: list[dict], namespace: str) -> None:
    for i in range(0, len(vectors), UPSERT_BATCH_SIZE):
        batch = vectors[i:i + UPSERT_BATCH_SIZE]
        upsert_fn(vectors=batch, namespace=namespace)


async def index_document(
    text, document_id, user_id, db, category,
    chunker_fn, embed_fn, upsert_fn, create_chunks_fn,
    public_ns_fn, user_ns_fn, sync_fn=None,
    law_name="", law_date="", clean_text=True, is_public=False,
    pre_chunked=None,
):
    filtered = _filter_text(text)
    if filtered is None:
        raise ValueError(f"[{document_id}] 텍스트가 너무 짧거나 비어있음")

    namespace = public_ns_fn() if is_public else user_ns_fn(str(user_id))

    if pre_chunked:
        chunks = pre_chunked
    elif is_public:
        chunks = chunker_fn(filtered, already_cleaned=True, clean_text=False)
    else:
        chunks = chunker_fn(filtered, clean_text=clean_text)

    if not chunks:
        raise ValueError(f"[{document_id}] 청킹 결과 없음")

    texts = [c.text for c in chunks]
    embedding_results = embed_fn(texts)

    if len(embedding_results) != len(chunks):
        raise ValueError(
            f"[{document_id}] 임베딩 수 불일치: 청크={len(chunks)}, 임베딩={len(embedding_results)}"
        )

    rdb_rows = create_chunks_fn(
        db=db,
        chunk_texts=[c.text for c in chunks],
        articles=[c.article for c in chunks],
        document_id=document_id,
        law_date=law_date,
    )

    chunk_ids = [str(row.vector_id) for row in rdb_rows]
    vectors = build_pinecone_vectors(
        chunks=chunks, chunk_ids=chunk_ids, document_id=str(document_id),
        category=category, law_name=law_name, law_date=law_date,
        embeddings=[e.dense for e in embedding_results],
        sparse_vectors=[e.sparse for e in embedding_results],
    )

    _upsert_in_batches(upsert_fn, vectors, namespace)

    synced = 0
    if not is_public and sync_fn:
        synced = await sync_fn(
            chunks=chunks, user_embs=embedding_results,
            category=category, law_name=law_name,
            user_law_date=law_date, db=db,
        )

    return IndexingResult(
        document_id=str(document_id),
        doc_type=category,
        total_chunks=len(chunks),
        namespace=namespace,
        synced_public_chunks=synced,
    )


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_embedding(dim=4):
    e = SimpleNamespace()
    e.dense = [0.1] * dim
    e.sparse = {"indices": [0], "values": [1.0]}
    return e

def _make_chunk(text="법 조항 텍스트입니다.", idx=0, article="제1조"):
    return Chunk(text=text, chunk_index=idx, article=article, law_name="")

def _make_rdb_row():
    row = MagicMock()
    row.vector_id = uuid.uuid4()
    return row


@pytest.fixture
def base_args():
    """index_document에 주입할 기본 의존성 모음"""
    chunks = [_make_chunk(idx=i) for i in range(3)]
    embs = [_make_embedding() for _ in chunks]
    rows = [_make_rdb_row() for _ in chunks]

    chunker_fn = MagicMock(return_value=chunks)
    embed_fn = MagicMock(return_value=embs)
    upsert_fn = MagicMock()
    create_chunks_fn = MagicMock(return_value=rows)
    db = MagicMock()
    sync_fn = AsyncMock(return_value=0)

    return dict(
        text="이것은 충분히 긴 법령 텍스트입니다. 테스트용 문서입니다.",
        document_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        db=db,
        category="법령·규정",
        law_name="도로교통법",
        law_date="2024-01-01",
        chunker_fn=chunker_fn,
        embed_fn=embed_fn,
        upsert_fn=upsert_fn,
        create_chunks_fn=create_chunks_fn,
        public_ns_fn=lambda: "public",
        user_ns_fn=lambda uid: f"user_{uid}",
        sync_fn=sync_fn,
        chunks=chunks,
        rows=rows,
    )


# ── _filter_text 테스트 ───────────────────────────────────────────────────────

class TestFilterText:
    def test_valid_text_passes(self):
        assert _filter_text("이 텍스트는 충분히 긴 문자열입니다.") is not None

    def test_short_text_returns_none(self):
        assert _filter_text("짧음") is None

    def test_exactly_20_chars(self):
        text = "가" * 20
        assert _filter_text(text) == text

    def test_19_chars_returns_none(self):
        assert _filter_text("가" * 19) is None

    def test_strips_whitespace_before_check(self):
        # 공백 제거 후 길이 체크
        assert _filter_text("   " + "가" * 19 + "   ") is None

    def test_only_whitespace_returns_none(self):
        assert _filter_text("          ") is None


# ── build_pinecone_vectors 테스트 ─────────────────────────────────────────────

class TestBuildPineconeVectors:
    def _call(self, chunks=None, law_name="근로기준법", category="법령·규정"):
        if chunks is None:
            chunks = [_make_chunk(idx=0, article="제1조")]
        ids = [str(uuid.uuid4()) for _ in chunks]
        embs = [[0.1] * 4 for _ in chunks]
        sparses = [{"indices": [0], "values": [1.0]} for _ in chunks]
        doc_id = str(uuid.uuid4())
        vectors = build_pinecone_vectors(
            chunks=chunks, chunk_ids=ids, document_id=doc_id,
            category=category, law_name=law_name, law_date="2024-01-01",
            embeddings=embs, sparse_vectors=sparses,
        )
        return vectors, ids, doc_id

    def test_vector_count_matches_chunks(self):
        chunks = [_make_chunk(idx=i) for i in range(5)]
        vectors, _, _ = self._call(chunks)
        assert len(vectors) == 5

    def test_vector_id_matches_chunk_id(self):
        vectors, ids, _ = self._call()
        assert vectors[0]["id"] == ids[0]
        assert vectors[0]["metadata"]["vector_id"] == ids[0]

    def test_document_id_in_metadata(self):
        vectors, _, doc_id = self._call()
        assert vectors[0]["metadata"]["document_id"] == doc_id

    def test_article_composed_with_law_name(self):
        chunk = Chunk(text="텍스트", chunk_index=0, article="제5조", law_name="")
        vectors, _, _ = self._call(chunks=[chunk], law_name="근로기준법")
        assert vectors[0]["metadata"]["article"] == "근로기준법 제5조"

    def test_law_name_fallback_to_chunk_law_name(self):
        chunk = Chunk(text="텍스트", chunk_index=0, article="제3조", law_name="형법")
        vectors, _, _ = self._call(chunks=[chunk], law_name="")
        assert "형법" in vectors[0]["metadata"]["article"]

    def test_law_name_fallback_to_category(self):
        chunk = Chunk(text="텍스트", chunk_index=0, article="제1조", law_name="")
        vectors, _, _ = self._call(chunks=[chunk], law_name="", category="도로교통")
        assert "도로교통법" in vectors[0]["metadata"]["article"]

    def test_no_article_uses_law_name_only(self):
        chunk = Chunk(text="텍스트", chunk_index=0, article="", law_name="")
        vectors, _, _ = self._call(chunks=[chunk], law_name="민법")
        assert vectors[0]["metadata"]["article"] == "민법"

    def test_metadata_has_required_keys(self):
        vectors, _, _ = self._call()
        meta = vectors[0]["metadata"]
        for key in ["vector_id", "document_id", "chunk_index", "article",
                    "category", "law_date", "created_at", "updated_at"]:
            assert key in meta, f"metadata에 '{key}' 없음"

    def test_chunk_index_in_metadata(self):
        chunks = [_make_chunk(idx=i) for i in range(3)]
        vectors, _, _ = self._call(chunks=chunks)
        for i, v in enumerate(vectors):
            assert v["metadata"]["chunk_index"] == i


# ── _upsert_in_batches 테스트 ─────────────────────────────────────────────────

class TestUpsertInBatches:
    def test_single_batch_when_under_limit(self):
        upsert_fn = MagicMock()
        vectors = [{"id": f"v{i}"} for i in range(10)]
        _upsert_in_batches(upsert_fn, vectors, "public")
        assert upsert_fn.call_count == 1

    def test_splits_into_multiple_batches(self):
        upsert_fn = MagicMock()
        # UPSERT_BATCH_SIZE=200, 250개 → 2번 호출
        vectors = [{"id": f"v{i}"} for i in range(250)]
        _upsert_in_batches(upsert_fn, vectors, "public")
        assert upsert_fn.call_count == 2

    def test_exact_batch_size(self):
        upsert_fn = MagicMock()
        vectors = [{"id": f"v{i}"} for i in range(200)]
        _upsert_in_batches(upsert_fn, vectors, "public")
        assert upsert_fn.call_count == 1

    def test_namespace_forwarded(self):
        upsert_fn = MagicMock()
        _upsert_in_batches(upsert_fn, [{"id": "v1"}], "user_abc")
        _, kwargs = upsert_fn.call_args
        assert kwargs["namespace"] == "user_abc"

    def test_all_vectors_sent(self):
        upsert_fn = MagicMock()
        vectors = [{"id": f"v{i}"} for i in range(350)]
        _upsert_in_batches(upsert_fn, vectors, "public")
        # 2번 호출: 200 + 150
        assert upsert_fn.call_count == 2
        sent = []
        for c in upsert_fn.call_args_list:
            sent.extend(c.kwargs["vectors"])
        assert len(sent) == 350

    def test_empty_vectors(self):
        upsert_fn = MagicMock()
        _upsert_in_batches(upsert_fn, [], "public")
        upsert_fn.assert_not_called()


# ── _build_article_key 테스트 ─────────────────────────────────────────────────

class TestBuildArticleKey:
    def test_law_name_and_article(self):
        chunk = Chunk(text="t", chunk_index=0, article="제3조", law_name="")
        assert _build_article_key(chunk, "형법", "형사") == "형법 제3조"

    def test_chunk_law_name_used_when_no_law_name(self):
        chunk = Chunk(text="t", chunk_index=0, article="제1조", law_name="민법")
        assert _build_article_key(chunk, "", "민사") == "민법 제1조"

    def test_category_fallback(self):
        chunk = Chunk(text="t", chunk_index=0, article="제2조", law_name="")
        assert _build_article_key(chunk, "", "형사") == "형사법 제2조"

    def test_no_article_returns_law_name_only(self):
        chunk = Chunk(text="t", chunk_index=0, article="", law_name="")
        assert _build_article_key(chunk, "근로기준법", "노동") == "근로기준법"

    def test_no_article_no_law_name_returns_category_fallback(self):
        chunk = Chunk(text="t", chunk_index=0, article="", law_name="")
        assert _build_article_key(chunk, "", "도로교통") == "도로교통법"


# ── index_document 통합 로직 테스트 ──────────────────────────────────────────

class TestIndexDocument:
    @pytest.mark.asyncio
    async def test_returns_indexing_result(self, base_args):
        args = base_args
        result = await index_document(**{k: v for k, v in args.items()
                                         if k not in ("chunks", "rows")})
        assert isinstance(result, IndexingResult)
        assert result.total_chunks == 3
        assert result.doc_type == "법령·규정"

    @pytest.mark.asyncio
    async def test_short_text_raises(self, base_args):
        args = dict(base_args)
        args["text"] = "짧음"
        with pytest.raises(ValueError, match="짧거나 비어있음"):
            await index_document(**{k: v for k, v in args.items()
                                    if k not in ("chunks", "rows")})

    @pytest.mark.asyncio
    async def test_empty_chunks_raises(self, base_args):
        args = dict(base_args)
        args["chunker_fn"] = MagicMock(return_value=[])
        with pytest.raises(ValueError, match="청킹 결과 없음"):
            await index_document(**{k: v for k, v in args.items()
                                    if k not in ("chunks", "rows")})

    @pytest.mark.asyncio
    async def test_embedding_count_mismatch_raises(self, base_args):
        args = dict(base_args)
        # 청크 3개인데 임베딩 2개
        args["embed_fn"] = MagicMock(return_value=[_make_embedding(), _make_embedding()])
        with pytest.raises(ValueError, match="임베딩 수 불일치"):
            await index_document(**{k: v for k, v in args.items()
                                    if k not in ("chunks", "rows")})

    @pytest.mark.asyncio
    async def test_user_namespace_used_when_not_public(self, base_args):
        args = dict(base_args)
        result = await index_document(**{k: v for k, v in args.items()
                                         if k not in ("chunks", "rows")},
                                       is_public=False)
        assert result.namespace.startswith("user_")

    @pytest.mark.asyncio
    async def test_public_namespace_used_when_public(self, base_args):
        args = dict(base_args)
        result = await index_document(**{k: v for k, v in args.items()
                                         if k not in ("chunks", "rows")},
                                       is_public=True)
        assert result.namespace == "public"

    @pytest.mark.asyncio
    async def test_rdb_insert_called(self, base_args):
        args = base_args
        await index_document(**{k: v for k, v in args.items()
                                 if k not in ("chunks", "rows")})
        args["create_chunks_fn"].assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_called(self, base_args):
        args = base_args
        await index_document(**{k: v for k, v in args.items()
                                 if k not in ("chunks", "rows")})
        assert args["upsert_fn"].call_count >= 1

    @pytest.mark.asyncio
    async def test_pre_chunked_skips_chunker(self, base_args):
        args = dict(base_args)
        pre = [_make_chunk(idx=0)]
        args["embed_fn"] = MagicMock(return_value=[_make_embedding()])
        args["create_chunks_fn"] = MagicMock(return_value=[_make_rdb_row()])

        result = await index_document(
            **{k: v for k, v in args.items() if k not in ("chunks", "rows")},
            pre_chunked=pre,
        )
        args["chunker_fn"].assert_not_called()
        assert result.total_chunks == 1

    @pytest.mark.asyncio
    async def test_sync_not_called_for_public_doc(self, base_args):
        args = base_args
        await index_document(**{k: v for k, v in args.items()
                                 if k not in ("chunks", "rows")},
                              is_public=True)
        args["sync_fn"].assert_not_called()

    @pytest.mark.asyncio
    async def test_sync_called_for_user_doc(self, base_args):
        args = base_args
        await index_document(**{k: v for k, v in args.items()
                                 if k not in ("chunks", "rows")},
                              is_public=False)
        args["sync_fn"].assert_called_once()

    @pytest.mark.asyncio
    async def test_synced_count_in_result(self, base_args):
        args = dict(base_args)
        args["sync_fn"] = AsyncMock(return_value=2)
        result = await index_document(**{k: v for k, v in args.items()
                                          if k not in ("chunks", "rows")})
        assert result.synced_public_chunks == 2