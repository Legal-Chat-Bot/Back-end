# BGE-M3 하이브리드 임베딩
# Dense (1024d) + Sparse (BM25) 동시 생성

import os
# CPU 환경 최적화
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"

from dataclasses import dataclass
from FlagEmbedding import BGEM3FlagModel
from typing import cast, TypedDict
import numpy as np

from app.core.config import settings


# ── 모델 전역 캐싱 (한 번만 로드) ────────────────────────
_model: BGEM3FlagModel | None = None

# 모델 설정
EMBEDDING_DEVICE = settings.EMBEDDING_DEVICE
EMBEDDING_BATCH = settings.EMBEDDING_BATCH

# ✅ sparse 노이즈 필터 임계값 (이 값 이하 토큰은 버림)
# BGE-M3 lexical_weights는 0에 가까운 토큰이 많아 노이즈가 됨
SPARSE_THRESHOLD = settings.SPARSE_THRESHOLD


def get_model() -> BGEM3FlagModel:
    global _model

    if _model is None:
        _model = BGEM3FlagModel(
            settings.EMBEDDING_MODEL,
            use_fp16=True,
            device=EMBEDDING_DEVICE,
        )

    return _model


# ── 임베딩 결과 데이터클래스 ──────────────────────────────
@dataclass
class EmbeddingResult:
    dense: list[float]   # 1024차원
    sparse: dict         # {"indices": [...], "values": [...]}
    text: str


# 클래스로 타입 정의
class EncodeOutput(TypedDict):
    dense_vecs: np.ndarray
    lexical_weights: list[dict]
    colbert_vecs: np.ndarray | None


# ── sparse 파싱 + 후처리 ──────────────────────────────────
def _parse_sparse(raw_sparse: dict) -> dict:
    """
    BGE-M3 lexical_weights → Pinecone sparse 포맷으로 변환
    1. SPARSE_THRESHOLD 이하 토큰 제거 (노이즈 필터)
    2. values 정규화 (max=1.0 기준) → dense와 스케일 맞춤
    """
    indices: list[int] = []
    values: list[float] = []

    if isinstance(raw_sparse, dict):
        if "indices" in raw_sparse and "values" in raw_sparse:
            # {"indices": [...], "values": [...]} 형태
            for idx, val in zip(raw_sparse["indices"], raw_sparse["values"]):
                if float(val) > SPARSE_THRESHOLD:  # ✅ 노이즈 필터
                    indices.append(int(idx))
                    values.append(float(val))
        else:
            # {"3412": 0.45, ...} 일반 형태
            for k, v in raw_sparse.items():
                if float(v) > SPARSE_THRESHOLD:    # ✅ 노이즈 필터
                    indices.append(int(k))
                    values.append(float(v))
    else:
        # dict-like 객체 안전 처리
        try:
            if hasattr(raw_sparse, "indices") and hasattr(raw_sparse, "values"):
                for idx, val in zip(
                    getattr(raw_sparse, "indices"),
                    getattr(raw_sparse, "values"),
                ):
                    if float(val) > SPARSE_THRESHOLD:
                        indices.append(int(idx))
                        values.append(float(val))
        except Exception:
            pass

    # ✅ sparse values 정규화 (max=1.0 기준)
    # dense는 BGE-M3가 L2 정규화하므로 sparse도 스케일 맞춰야 기여도 안정적
    if values:
        max_val = max(values)
        if max_val > 0:
            values = [v / max_val for v in values]

    return {"indices": indices, "values": values}


# ── 배치 임베딩 ───────────────────────────────────────────
def embed_texts(texts: list[str]) -> list[EmbeddingResult]:
    if not texts:
        return []

    model = get_model()

    raw = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    outputs = cast(EncodeOutput, raw)

    results = []
    for i, text in enumerate(texts):
        dense = outputs["dense_vecs"][i].tolist()
        raw_sparse = cast(dict, outputs["lexical_weights"][i])
        sparse = _parse_sparse(raw_sparse)  # ✅ 분리된 함수로 처리

        results.append(EmbeddingResult(dense=dense, sparse=sparse, text=text))

    return results


# ── 단일 텍스트 임베딩 (검색용) ──────────────────────────
def embed_query(query: str) -> EmbeddingResult:
    return embed_texts([query])[0]
