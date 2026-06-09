#인덱싱 파이프라인
#문서 → 거름망 → 청킹 → DB 저장(chunk_id 확보) → 임베딩 → Pinecone upsert

# 외부라이브러리
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID,uuid4

# vector 부분
import app.db.vector.client as pinecone
from app.db.vector.chunker import Chunker, ChunkConfig, Chunk

# 클래스를 좀더 가독성좋게 사용할수있게해주는 dataclass입니다.
@dataclass
class IndexingResult:
    # 보통 __init__들어가나 여기선 생략이됩니다.
    document_id: str #원래는 self.document_id=document_id이런식으로 작성을 해야합니다.
    doc_type: str
    total_chunks: int
    namespace: str

    # 청킹 초기화 부분(전역, 한번만)
_chunker = Chunker(ChunkConfig(
    max_chars=500,
    overlap_chars=50,
    min_chars=10,
))    

# ── 텍스트 거름망 ─────────────────────────────────────────
def _filter_text(text: str) -> str | None:
    """너무 짧거나 의미 없는 텍스트 걸러냄"""
    text = text.strip()
    if len(text) < 20:
        return None
    return text

def build_pincone_vectors(
        chunks: list[Chunk],
        chunk_ids: list[str],   # RDB에서 발급된 chunk_id 리스트
        category:str,
        law_date:str,
        embeddings:list[list[float]],
        sparse_vectors: list[dict],
)-> list[dict]:
    '''
    청크 + 임베딩 -> Pincone upsert용 벡터 리스트 조립
    '''
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    vectors = []

    for chunk, chunk_id, dense, sparse in zip(chunks, chunk_ids, embeddings, sparse_vectors):
        vectors.append({
            "id": str(uuid4()),          # RDB chunk_id를 vector_id로 사용
            "values": dense,
            "sparse_values": sparse,
            "metadata": {
                "vector_id":  chunk_id,   # RDB chunks 테이블 id 참조용
                "category":   category,
                "created_at": now,
                "updated_at": now,
                "law_date":   law_date,
            },
        })
    return vectors


# 법률 문서 인덱싱.
# 비동기 처리한다는것 async 아직 미구현
async def index_public_document(
    text: str,
    document_id: UUID,
    category: str,
    law_date: str,
) -> IndexingResult:
    namespace = pinecone.public_namespace()  # public이니까

    # 1. 거름망
    filtered = _filter_text(text)
    if filtered is None:
        raise ValueError(f"[{document_id}] 텍스트가 너무 짧거나 비어있음")
    
    # 2. 청킹
    chunks = _chunker.chunk(filtered, already_cleaned=True)
    if not chunks:
        raise ValueError(f"[{document_id}] 청킹 결과 없음")
    
    return IndexingResult(
        document_id=str(document_id),
        doc_type=category,
        total_chunks=len(chunks),
        namespace=namespace,
    )

# 유저가 올린파일 인덱싱# ── 유저 문서 인덱싱 ──────────────────────────────────────
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

    # 3~5. 미구현 (index_public_document 참고)

    return IndexingResult(
        document_id=str(document_id),
        doc_type=category,
        total_chunks=len(chunks),
        namespace=namespace,
    )



