from app.core.config import settings
from pinecone import Pinecone, ServerlessSpec
from pinecone.data import Index

from uuid import UUID


# _pc ← 언더스코어 = 모듈 내부에서만 쓰는 변수라는 관례 =>해당모듈에서만사용.
# _index와 마찬가지로 pinecone같은변수 구분용.
_pc: Pinecone = None #초기값 None
# 실제 인덱스 객체 ("index명" 인덱스에 대한 핸들)
# Index 타입이거나, 아직 연결 안 했으면 None 
_index: Index  = None #마찬가지 초기값 None


# 인덱스 연결
def get_index() :
    global _pc, _index #전역변수 사용선언.=>함수끝나도 해당 변수 사용할수있게하기위함.

    if _pc is None:
        _pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    
    assert _pc is not None   # 문법상 "여기서부터 _pc는 None 아님" 보장

    # 인덱스가 존재하면 바로연결 아니면 생성    
    if _index is None:
        # 1. [수정] index_exists() 대신 list_indexes()를 사용하여 존재 여부 확인 (최신 API 문법)
        active_indexes = [idx.name for idx in _pc.list_indexes()]
        
        if settings.PINECONE_INDEX_NAME not in active_indexes:
            _pc.create_index(
                name=settings.PINECONE_INDEX_NAME,
                dimension=1024,
                metric="dotproduct",
                spec=ServerlessSpec(
                    cloud=settings.PINECONE_CLOUD,
                    region=settings.PINECONE_REGION,
                ),
            )
        _index = _pc.Index(settings.PINECONE_INDEX_NAME)
    
    return _index


# 네임스페이스로 구분 헬퍼
def public_namespace() -> str:
    return "public"

def user_namespace(user_id: UUID) -> str:
    return f"user_{user_id}"

# Upsert
def upsert(vectors: list[dict], namespace:str)-> dict:
    return get_index().upsert(vectors=vectors, namespace=namespace)

# Query
def query(dense_vector: list[float], sparse_vector:dict, namespace:str, top_k:int=5, filter: dict | None=None, alpha:float=0.3) -> list[dict]:
    # dense_vector값과 곱해주는 함수입니다. v가 dense_vector값 이는 70퍼 할당한다는 의미입니다.
    # sparse_vector는 자동으로 30퍼만할당.
    scaled_dense = [v * alpha for v in dense_vector]
    scaled_sparse = {
        # Sparse 벡터에서 어떤 토큰(단어)인지 가리키는 번호. indices랑 values하나가 쌍.
        "indices": sparse_vector["indices"],
        "values": [v * (1-alpha) for v in sparse_vector["values"]],
    }
    
    response = get_index().query(
        vector=scaled_dense,
        sparse_vector=scaled_sparse,
        namespace=namespace,
        top_k=top_k,
        filter=filter,
        include_metadata=True,
    )
    
    # 2. [수정] 최신 Pinecone SDK의 QueryResponse 객체는 딕셔너리처럼 .get()을 쓰면 에러가 날 수 있습니다.
    # 안전하게 점 접근 방식(response.matches)을 쓰거나 객체 구조에 맞춰 처리해야 합니다.
    if hasattr(response, "matches"):
        return [match.to_dict() for match in response.matches]
    return response.get("matches", [])

# 삭제 로직 document_id값 기준 지울지 고민
def delete_by_document_id(document_id:str, namespace:str) -> None:
    get_index().delete(
        # eq pinecone에서 equal의 의미 같다는 의미로사용함..
        filter={"document_id": {"$eq": document_id}},
        namespace=namespace,
    )

# ✅ 네임스페이스 전체 초기화 (재인덱싱 전 중복 제거용)
def delete_all(namespace: str) -> None:
    get_index().delete(delete_all=True, namespace=namespace)
    print(f"[Pinecone] 전체 삭제 완료: namespace={namespace}")

def delete_by_ids(vector_ids: list[str], namespace: str) -> None:
    """
    RDB에서 수집한 vector_id 목록으로 Pinecone 벡터를 직접 삭제.
    vector_ids가 비어 있으면 아무것도 하지 않는다 (Pinecone API 에러 방지).
    """
    if not vector_ids:
        return
    get_index().delete(ids=vector_ids, namespace=namespace)
