#인덱싱 파이프라인
#문서 → 거름망 → 청킹 → DB 저장(chunk_id 확보) → 임베딩 → Pinecone upsert

# 외부라이브러리
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID
# vector 부분
import app.db.vector.client as pinecone
# from app.db.vector.chunker
# from app.db.vector.embedding
# from app.db.vector.filter

# 클래스를 좀더 가독성좋게 사용할수있게해주는 dataclass입니다.
@dataclass
class IndexingResult:
    # 보통 __init__들어가나 여기선 생략이됩니다.
    document_id: str #원래는 self.document_id=document_id이런식으로 작성을 해야합니다.
    doc_type: str
    total_chunks: int
    namespace: str


def build_pincone_vectors():
    return ""


# 법률 문서 인덱싱.
# 비동기 처리한다는것 async 아직 미구현
# async def index_public_document(
#     text:str,
#     documnet_id:UUID,
#     vector_id:UUID,
#     category: str,
# ) -> IndexingResult:
#     # 공용인덱스 함수불러옴.
#     namespace = pincone.public_namespace()

#     # 1.거름망
#     filtered =

#     # 2.청킹
#     chunks=

#     # 3.rdb저장
#     db_chunks

#     # 4.임베딩

#     5.pinconeupsert

#     return IndexingResult


# 유저가 올린파일 인덱싱
# async def inedx_user_document()



