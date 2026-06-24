# ============================================================
# 인덱싱 파이프라인
#
# 문서 → 거름망 → 청킹 → 임베딩 → Pinecone upsert
#
# 개선 사항:
#   - Pinecone 메타데이터에 article(조) 추가
#   - 청크 수 / 임베딩 수 불일치 검증
#   - law_name 없을 때 chunk.law_name(텍스트 추출) → category+"법" 순으로 폴백
#   - index_user_document에 law_date 추가
#   - 유저 문서 law_date가 공용보다 최신이면 공용 DB text/law_date/updated_at 동기화
#   - index_user_document에 clean_text 파라미터 추가
#       유저가 업로드하는 문서가 법률 구조(조)를 가진 문서인지 여부에 따라
#       chunker의 조(article) 탐지를 켜고 끌 수 있도록 함.
#       · clean_text=True  (기본값) → 조 탐지 수행 (법령, 판례 등 법률 문서)
#       · clean_text=False → 조 탐지 스킵 (회의록, 매뉴얼 등 일반 문서)
#     공용(법률) 문서는 항상 조 구조를 가지므로 index_public_document는 True로 고정.
# ============================================================

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy.orm import Session
from fastapi import Depends

from app.db.db import get_db
import app.db.vector.client as pinecone
from app.db.vector.embedding import embed_texts
from app.db.vector.chunker import Chunker, Chunk
from app.crud.chunk_crud import (
    get_chunks_by_document,
    delete_chunks_from_rdb,
    create_chunks_bulk,
)
from app.crud.chunk_dataset_crud import update_chunk_dataset_text
from app.db.models.document import Document





UPSERT_BATCH_SIZE = 200  # Pinecone 권장값

# 배치 upsert 한번에 파일이 올라가면 오류가 발생하기에 방지용
def _upsert_in_batches(vectors: list[dict], namespace: str) -> None:
    for i in range(0, len(vectors), UPSERT_BATCH_SIZE):
        batch = vectors[i:i + UPSERT_BATCH_SIZE]
        pinecone.upsert(vectors=batch, namespace=namespace)


@dataclass
class IndexingResult:
    document_id: str
    doc_type: str
    total_chunks: int
    namespace: str
    synced_public_chunks: int = 0  # 공용 DB에 동기화된 청크 수


# ── 청커 초기화 (ChunkConfig 기본값 사용) ──────────────────
_chunker = Chunker()


# ── 텍스트 거름망 ──────────────────────────────────────────
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
    document_id: str,           # 문서마다 청크가 다르니 조회를 위해 넣음.
    category: str,
    law_name: str,
    law_date: str,
    embeddings: list[list[float]],
    sparse_vectors: list[dict],
) -> list[dict]:
    """
    청크 + 임베딩 → Pinecone upsert용 벡터 리스트 조립

    메타데이터:
      - vector_id    : RDB chunks 테이블 PK (= Pinecone 벡터 id와 동일)
      - document_id  : RDB document 테이블 FK (필터 삭제용)
      - chunk_index  : 문서 내 청크 순서
      - article      : 법령명 + 조  예) "근로기준법 제3조"
      - category     : 문서 카테고리
      - law_date     : 법령 시행일 (유저 문서는 빈 문자열)
      - created_at / updated_at
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    vectors = []

    for chunk, chunk_id, dense, sparse in zip(chunks, chunk_ids, embeddings, sparse_vectors):
        # article 조립 우선순위:
        #   1순위: 인자로 받은 law_name (호출 시 명시적으로 전달된 경우)
        #   2순위: 청크 텍스트에서 추출한 chunk.law_name
        #   3순위: category + "법" 폴백
        resolved_law_name = (
            law_name
            or chunk.law_name
            or f"{category}법"
        )
        if chunk.article:
            article = f"{resolved_law_name} {chunk.article}".strip()
        else:
            article = resolved_law_name
        # RDB vector_id와 동일하게 맞춰야 삭제 시 RDB → Pinecone ID 매핑이 성립함
        vectors.append({
            "id": chunk_id,
            "values": dense,
            "sparse_values": sparse,
            "metadata": {
                "vector_id":   chunk_id,
                "document_id": document_id,   # 문서마다 청크 조회겸 삭제
                "chunk_index": chunk.chunk_index, #몇번째 청크인지.
                "article":     article,
                "category":    category,
                "law_date":    law_date,
                "created_at":  now,
                "updated_at":  now,
            },
        })

    return vectors


# ── article_key 조립 헬퍼 ────────────────────────────────
def _build_article_key(chunk: Chunk, law_name: str, category: str) -> str:
    resolved = law_name or chunk.law_name or f"{category}법"
    return f"{resolved} {chunk.article}".strip() if chunk.article else resolved


# ── 공용 DB 동기화 ────────────────────────────────────────
async def _sync_public_if_newer(
    chunks: list[Chunk],
    user_embs: list,           # index_user_document에서 이미 만든 임베딩 재사용
    category: str,
    law_name: str,
    user_law_date: str,
    db: Session,  #RDB chunk_text 갱신용.  
) -> int:
    """
    유저 문서의 law_date가 공용 DB보다 최신이면
    내용이 유사한 공용 벡터의 text / law_date / updated_at을 갱신한다.

    매칭 전략 (유저 청크 단위):
      1. 유저 청크 임베딩으로 공용 DB 쿼리 (article 필터 + 유사도)
      2. 최상위 매칭 공용 벡터의 law_date 비교
      3. 유저가 더 최신이고 유사도 임계값(SCORE_THRESHOLD) 이상이면
         해당 공용 벡터를 유저 text로 교체 후 upsert
      4. 한 공용 벡터는 한 번만 갱신 (중복 갱신 방지)

    반환: 실제로 갱신된 공용 벡터 수
    """
    if not user_law_date:
        return 0

    # 유사도 임계값: 이 점수 미만이면 다른 내용으로 판단해 스킵
    SCORE_THRESHOLD = 0.75

    pub_namespace = pinecone.public_namespace()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    synced = 0
    updated_pub_ids: set[str] = set()  # 동일 공용 벡터 중복 갱신 방지

    update_vectors: list[dict] = []

    for chunk, emb in zip(chunks, user_embs):
        article_key = _build_article_key(chunk, law_name, category)

        # 유저 청크 임베딩 + article 필터로 공용 DB에서 가장 유사한 청크 검색
        matches = pinecone.query(
            dense_vector=emb.dense,
            sparse_vector=emb.sparse,
            namespace=pub_namespace,
            top_k=1,
            filter={"article": {"$eq": article_key}},
        )
        if not matches:
            continue

        best = matches[0]
        score        = best.get("score", 0.0)
        pub_id       = best.get("id", "")
        pub_meta     = best.get("metadata", {})
        pub_law_date = pub_meta.get("law_date", "")

        # 유사도 임계값 미달 → 다른 내용의 청크로 판단, 스킵
        if score < SCORE_THRESHOLD:
            continue

        # 공용이 이미 같거나 최신 → 스킵
        if pub_law_date and pub_law_date >= user_law_date:
            continue

        # 이미 이번 배치에서 갱신 예정인 공용 벡터 → 중복 스킵
        if pub_id in updated_pub_ids:
            continue

        updated_pub_ids.add(pub_id)

        # 유저 청크 text로 공용 벡터 갱신 (임베딩은 이미 보유)
        new_meta = dict(pub_meta)
        new_meta["law_date"]   = user_law_date
        new_meta["updated_at"] = now

        update_vectors.append({
            "id":            pub_id,       # 기존 공용 벡터 ID 유지
            "values":        emb.dense,    # 유저 청크 임베딩 재사용 (재임베딩 불필요)
            "sparse_values": emb.sparse,
            "metadata":      new_meta,
        })

        # RDB: chunk_text / law_date 교체 (pub_id == vector_id로 직접 매핑)
        update_chunk_dataset_text(
            db=db,
            vector_id=UUID(pub_id),
            new_text=chunk.text,
            new_law_date=user_law_date,
        )

    if update_vectors:
        _upsert_in_batches(update_vectors, pub_namespace)
        synced = len(update_vectors)

    return synced

#문서 삭제
async def delete_document_index(
        document: Document,
        db:Session,
) -> int:
    '''
    문서 인덱스 전체 삭제
    삭제 되는 순서(Vector DB -> RDB순서)
        1. RDB에서 document_id에 속한 모든 Chunk를 조회 vector_id 목록을 수집
        => Pincone 백터 ID와 RDB PK가 동일하므로 직접 매핑이 가능함.
        2. Pinecone에서 해당 vector_id 목록을 먼저 삭제.
        -> 이 시점까지 RDB행은 살아있음. 오류 발생시 재시도 가능함.
        3. RDB에서 Chunk 행삭제.
    '''
    #1. RDB에서 vector_id 목록 수집.
    chunks = get_chunks_by_document(db, document.document_id)
    if not chunks:
        return 0 # 하나도 조회 안되었다는뜻.

    vector_ids = [str(chunk.vector_id) for chunk in chunks]

    namespace=pinecone.user_namespace(str(document.user_id))
    print(document.user_id)
    # 2. Pinecone 먼저 삭제 (Vector DB -> RDB 순)
    pinecone.delete_by_ids(vector_ids, namespace=namespace)

    return document



#두개 로직이 겺쳐서 하나로 합칩니다. 혹시 에러가 날수도 있으니 아직 원래 코드 남깁니다. 공용함수 에러생기면
#두개 함수로 하겠습니다.
# ── 통합 문서 인덱싱 ──────────────────────────────────────
async def index_document(
    text: str,
    document_id: UUID,
    user_id: UUID,           
    db: Session,
    category: str,
    law_name: str = "",
    law_date: str = "",
    clean_text: bool = True,
    is_public: bool = False,  # 공용 문서(관리자 업로드) 여부 플래그
    pre_chunked: list[Chunk] | None = None,
) -> IndexingResult:
    """
    통합 문서 인덱싱 파이프라인 (공용/유저 문서 공용)
    """
    # 1. 거름망
    filtered = _filter_text(text)
    if filtered is None:
        raise ValueError(f"[{document_id}] 텍스트가 너무 짧거나 비어있음")
    
    namespace = pinecone.public_namespace() if is_public else pinecone.user_namespace(user_id)
    # 2. 문서 유형(공용/유저)에 따른 네임스페이스 및 청킹 분기 (원래 방식으로 안전하게 호출)
    if pre_chunked:
        chunks = pre_chunked
    elif is_public:
        # 공용 문서는 항상 법률 구조 분리 활성화 + 정제 스킵
        chunks = _chunker.chunk(filtered, already_cleaned=True, clean_text=False)
    else:
        # 유저 문서는 파라미터로 받은 clean_text 옵션 그대로 적용
        chunks = _chunker.chunk(filtered, clean_text=clean_text)


    # 3. 청킹 결과 검증
    if not chunks:
        raise ValueError(f"[{document_id}] 청킹 결과 없음")

    # 4. 임베딩 생성
    texts = [c.text for c in chunks]
    embedding_results = embed_texts(texts)

    # 5. [버그 수정] RDB 저장 전 임베딩 수 검증을 먼저 수행하여 데이터 정합성 보장
    if len(embedding_results) != len(chunks):
        raise ValueError(
            f"[{document_id}] 임베딩 수 불일치: 청크={len(chunks)}, 임베딩={len(embedding_results)}"
        )

    # 6. RDB Bulk Insert → vector_id 확보 (안전하게 검증 완료 후 커밋)
    rdb_rows = create_chunks_bulk(
        db=db,
        chunk_texts=[c.text for c in chunks],
        articles=[c.article for c in chunks],
        document_id=document_id,
        law_date=law_date,
    )

    # 7. Pinecone 벡터 조립
    chunk_ids = [str(row.vector_id) for row in rdb_rows]
    vectors = build_pinecone_vectors(
        chunks=chunks,
        chunk_ids=chunk_ids,
        document_id=str(document_id),
        category=category,
        law_name=law_name,
        law_date=law_date,
        embeddings=[e.dense for e in embedding_results],
        sparse_vectors=[e.sparse for e in embedding_results],
    )

    # 8. Pinecone 배치 Upsert
    _upsert_in_batches(vectors, namespace)

    # 9. 공용 DB 동기화 (유저 문서 업로드 시에만 작동)
    synced = 0
    if not is_public:
        synced = await _sync_public_if_newer(
            chunks=chunks,
            user_embs=embedding_results,
            category=category,
            law_name=law_name,
            user_law_date=law_date,
            db=db,
        )

    return IndexingResult(
        document_id=str(document_id),
        doc_type=category,
        total_chunks=len(chunks),
        namespace=namespace,
        synced_public_chunks=synced,
    )


# # ── 공용(법률) 문서 인덱싱 ────────────────────────────────
# async def index_public_document(
#     text: str,
#     document_id: UUID,
#     user_id:UUID,
#     session_id: UUID,   # rdb연동
#     db: Session,        # rdb연동
#     category: str,
#     law_name: str,
#     law_date: str,
# ) -> IndexingResult:
#     namespace = pinecone.public_namespace()

#     # 1. 거름망
#     filtered = _filter_text(text)
#     if filtered is None:
#         raise ValueError(f"[{document_id}] 텍스트가 너무 짧거나 비어있음")

#     # 2. 청킹 (공용 문서는 법률 구조 분리 활성화 + 정제 스킵)
#     # article이 이미 존재하는 공용문서는 False처리
#     chunks = _chunker.chunk(filtered, already_cleaned=True, clean_text=False)
#     if not chunks:
#         raise ValueError(f"[{document_id}] 청킹 결과 없음")

#     # 3. 임베딩
#     texts = [c.text for c in chunks]
#     embedding_results = embed_texts(texts)

#     if len(embedding_results) != len(chunks):
#         raise ValueError(
#             f"[{document_id}] 임베딩 수 불일치: 청크={len(chunks)}, 임베딩={len(embedding_results)}"
#         )
    
#     # 3. RDB 먼저 insert → vector_id 확보
#     rdb_rows = create_chunks_bulk(
#         db=db,
#         chunk_texts=[c.text for c in chunks],
#         articles=[c.article for c in chunks],
#         document_id=document_id,
#         session_id=session_id,
#         user_id=user_id,
#         law_date=law_date,
#     )

#     # 4. Pinecone 벡터 조립
#     chunk_ids = [str(row.vector_id) for row in rdb_rows]
#     vectors = build_pinecone_vectors(
#         chunks=chunks,
#         chunk_ids=chunk_ids,
#         document_id=str(document_id),
#         category=category,
#         law_name=law_name,
#         law_date=law_date,
#         embeddings=[e.dense for e in embedding_results],
#         sparse_vectors=[e.sparse for e in embedding_results],
#     )

#     # 5. Pinecone upsert
#     _upsert_in_batches(vectors, namespace)

#     return IndexingResult(
#         document_id=str(document_id),
#         doc_type=category,
#         total_chunks=len(chunks),
#         namespace=namespace,
#     )


# # ── 유저 문서 인덱싱 ──────────────────────────────────────
# async def index_user_document(
#     text: str,
#     document_id: UUID,
#     user_id:UUID,
#     session_id: UUID,   # rdb연동
#     db: Session,        # rdb연동
#     category: str,
#     law_name: str = "",   # 법률 문서면 "근로기준법" 등, 일반 문서면 빈 문자열
#     law_date: str = "",   # 법령 시행일  예) "2024-01-01", 일반 문서면 빈 문자열
#     clean_text: bool = True,
# ) -> IndexingResult:
    
#     namespace = pinecone.user_namespace(user_id)

#     # 1. 거름망
#     filtered = _filter_text(text)
#     if filtered is None:
#         raise ValueError(f"[{document_id}] 텍스트가 너무 짧거나 비어있음")

#     # 2. 청킹 (유저 문서는 정제 포함, 법률 구조 분리는 있으면 활용), clean_text를 chunker에 실제로 전달
#     chunks = _chunker.chunk(filtered, clean_text=clean_text)
#     if not chunks:
#         raise ValueError(f"[{document_id}] 청킹 결과 없음")

#     # 3. 임베딩
#     texts = [c.text for c in chunks]
#     embedding_results = embed_texts(texts)
#     if len(embedding_results) != len(chunks):
#         raise ValueError(
#             f"[{document_id}] 임베딩 수 불일치: 청크={len(chunks)}, 임베딩={len(embedding_results)}"
#         )

#     # 3.5 RDB 먼저 insert → vector_id 확보
#     rdb_rows = create_chunks_bulk(
#         db=db,
#         chunk_texts=[c.text for c in chunks],
#         articles=[c.article for c in chunks],
#         document_id=document_id,
#         session_id=session_id,
#         user_id=user_id,
#         law_date=law_date,
#     )


#     # 4. Pinecone 벡터 조립
#     # RDB가 생성한 vector_id를 그대로 Pinecone ID로 사용
#     chunk_ids = [str(row.vector_id) for row in rdb_rows]
#     vectors = build_pinecone_vectors(
#         chunks=chunks,
#         chunk_ids=chunk_ids,
#         document_id=str(document_id), 
#         category=category,
#         law_name=law_name,
#         law_date=law_date,
#         embeddings=[e.dense for e in embedding_results],
#         sparse_vectors=[e.sparse for e in embedding_results],
#     )

#     # 5. Pinecone upsert (유저 네임스페이스)
#     _upsert_in_batches(vectors, namespace)

#     # 6. 공용 DB 동기화: 유저 law_date가 공용보다 최신이면 공용 벡터 갱신
#     # embedding_results 재사용 → 추가 임베딩 비용 없음
#     # clean_text=False로 들어온 일반 문서는 law_date비우기.
#     # 호출하는 쪽에서 law_date 자체를 비워서 넘기는 것을 권장합니다.
#     synced = await _sync_public_if_newer(
#         chunks=chunks,
#         user_embs=embedding_results,
#         category=category,
#         law_name=law_name,
#         user_law_date=law_date,
#     )

#     return IndexingResult(
#         document_id=str(document_id),
#         doc_type=category,
#         total_chunks=len(chunks),
#         namespace=namespace,
#         synced_public_chunks=synced,
#     )