from __future__ import annotations

import hashlib
import json
import os
import sys

#절대경로설정 프로젝트폴더에서 실행.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from datetime import datetime
from FlagEmbedding import BGEM3FlagModel

from app.db.db import SessionLocal
from app.db.models.chunk_dataset import ChunkDataset
from app.db.vector.chunker import Chunker, ChunkConfig
import app.db.vector.client as pinecone
from app.crud.chunk_dataset_crud import create_chunk_dataset
from app.core.config import settings

# 상수 
UPSERT_BATCH     = 100
EMBED_BATCH      = 32     

# 임베딩 모델
_model: BGEM3FlagModel | None = None


def _get_model() -> BGEM3FlagModel:
    global _model
    if _model is None:
        print("[임베딩] 모델 로딩 중...")
        _model = BGEM3FlagModel(
            settings.EMBEDDING_MODEL,
            use_fp16=True,
            device="cuda",
        )
        print("[임베딩] 모델 로드 완료")
    return _model


def _parse_sparse(raw: dict) -> dict:
    indices, values = [], []
    if isinstance(raw, dict):
        if "indices" in raw and "values" in raw:
            for idx, val in zip(raw["indices"], raw["values"]):
                if float(val) > settings.SPARSE_THRESHOLD:
                    indices.append(int(idx))
                    values.append(float(val))
        else:
            for k, v in raw.items():
                if float(v) > settings.SPARSE_THRESHOLD:
                    indices.append(int(k))
                    values.append(float(v))
    else:
        try:
            if hasattr(raw, "indices") and hasattr(raw, "values"):
                for idx, val in zip(getattr(raw, "indices"), getattr(raw, "values")):
                    if float(val) > SPARSE_THRESHOLD:
                        indices.append(int(idx))
                        values.append(float(val))
        except Exception:
            pass

    if values:
        max_val = max(values)
        if max_val > 0:
            values = [v / max_val for v in values]

    return {"indices": indices, "values": values}


@dataclass
class EmbeddingResult:
    dense: list[float]
    sparse: dict
    text: str


def _embed_texts(texts: list[str]) -> list[EmbeddingResult]:
    if not texts:
        return []
    model = _get_model()
    raw = model.encode(
        texts,
        batch_size=EMBED_BATCH,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    results = []
    for i, text in enumerate(texts):
        dense  = raw["dense_vecs"][i].tolist()
        sparse = _parse_sparse(raw["lexical_weights"][i])
        results.append(EmbeddingResult(dense=dense, sparse=sparse, text=text))
    return results


# ── 메타데이터 추출 ───────────────────────────────────────



def _safe(value: object) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    return "" if s.lower() in "none" else s


def _extract_meta(record: dict) -> dict:
    raw_info = record.get("info")
    info: dict = raw_info if isinstance(raw_info, dict) else {}
    return {
        "article":  _safe(record.get("article")),
        "law_name": _safe(record.get("law_name")),
        "text":     _safe(record.get("source_text")),
        "law_date": (
            _safe(info.get("effectDate"))
            or _safe(info.get("finalDate"))
            or _safe(info.get("interpreDate"))
            or _safe(record.get("law_date"))
        ),
        "category": (
            _safe(info.get("agenda"))
            or _safe(record.get("category"))
        ),
    }


# ── 메인 적재 함수 ────────────────────────────────────────
def upsert_dataset(file_path: str):
    # 코랩과 동일한 청킹 설정
    chunker = Chunker(ChunkConfig(max_chars=1600, min_chars=30))
    db      = SessionLocal()

    processed_hashes: set[str] = set()
    total_chunks = skipped = 0

    try:
        with open(file_path, encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    doc = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[{idx}] 파싱 실패: {e}")
                    skipped += 1
                    continue

                meta     = _extract_meta(doc)
                article  = meta["article"]
                text     = meta["text"]
                law_name = meta["law_name"]
                law_date = meta["law_date"]

                # 스킵 조건
                if not article or not text or len(text.strip()) < 20:
                    skipped += 1
                    continue

                # 중복 텍스트 스킵 => 텍스트길이가 길어서 메모리 절약을 위해서 md5해시로 32자로 고정시킴.
                text_hash = hashlib.md5(text.strip().encode()).hexdigest()
                if text_hash in processed_hashes:
                    skipped += 1
                    continue
                processed_hashes.add(text_hash)

                # 청킹 (조 탐지 스킵, article은 메타데이터에서 가져온 값 사용)
                chunks = chunker.chunk(text, clean_text=False)
                if not chunks:
                    skipped += 1
                    continue

                # article 메타데이터 덮어씌우기
                for chunk in chunks:
                    chunk.article = article

                # 임베딩
                try:
                    embeddings = _embed_texts([c.text for c in chunks])
                except Exception as e:
                    print(f"[{idx}] 임베딩 오류: {e}")
                    skipped += 1
                    continue

                if len(embeddings) != len(chunks):
                    print(f"[{idx}] 불일치: 청크={len(chunks)}, 임베딩={len(embeddings)}")
                    skipped += 1
                    continue

                # RDB 먼저 → vector_id 확보
                try:
                    rdb_rows = create_chunk_dataset(
                        db=db,
                        chunk_texts=[c.text for c in chunks],
                        articles=[c.article for c in chunks],
                        law_date=law_date,
                    )
                except Exception as e:
                    print(f"[{idx}] RDB 오류: {e}")
                    skipped += 1
                    continue

                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Pinecone 벡터 조립
                vectors = [
                    {
                        "id":            str(row.dataset_vector_id),
                        "values":        emb.dense,
                        "sparse_values": emb.sparse,
                        "metadata": {
                            "vector_id": str(row.dataset_vector_id),
                            "article":   chunk.article,
                            "law_name": law_name,
                            "category":  meta["category"],
                            "law_date":  law_date,
                            "created_at":  now,
                            "updated_at":  now,
                        },
                    }
                    for chunk, row, emb in zip(chunks, rdb_rows, embeddings)
                ]

                # Pinecone upsert
                try:
                    for i in range(0, len(vectors), UPSERT_BATCH):
                        pinecone.upsert(vectors=vectors[i:i + UPSERT_BATCH], namespace=pinecone.public_namespace())
                    total_chunks += len(chunks)
                    print(f"[{idx}] {article} | chunks={len(chunks)} → 완료")
                except Exception as e:
                    print(f"[{idx}] Pinecone 오류: {e}")
                    # RDB 롤백 (방금 삽입한 행만 삭제)
                    for row in rdb_rows:
                        db.query(ChunkDataset).filter(
                            ChunkDataset.dataset_vector_id == row.dataset_vector_id
                        ).delete(synchronize_session=False)
                    db.commit()
                    skipped += 1
                    continue

    finally:
        db.close()

    print(f"\n[완료] 총 청크={total_chunks} | 스킵={skipped}")


if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "dataset/legal.jsonl"
    upsert_dataset(file_path)