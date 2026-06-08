from core.config import settings
from pinecone import Pinecone, ServerlessSpec
from pinecone.data import Index


# _pc ← 언더스코어 = 모듈 내부에서만 쓰는 변수라는 관례 =>해당모듈에서만사용.
# _ index와 마찬가지로 pinecone같은변수 구분용.
# pc ← 언더스코어 없으면 외부에서 자유롭게 접근 가능 
_pc: Pinecone | None = None #초기값 None
# 실제 인덱스 객체 ("index명" 인덱스에 대한 핸들)
# Index 타입이거나, 아직 연결 안 했으면 None 
# _ index중복 변수 구분용.
_index: Index | None = None #마찬가지 초기값 None


# 인덱스 연결
def get_index() ->Index:
    global _pc, _index #전역변수 사용선언.=>함수끝나도 해당 변수 사용할수있게하기위함.

    if _pc is None:
        _pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    
    assert _pc is not None   # 문법상 "여기서부터 _pc는 None 아님" 보장

    #인덱스가 존재하면 바로연결 아니면 생성    
    if _index is None:
        if not _pc.index_exists(settings.PINECONE_INDEX_NAME):
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

def user_namespace(user_id: str) -> str:
    return f"user_{user_id}"

# Upsert
def upsert(vectors: list[dict], namespace:str)-> dict:
    return get_index().upsert(vectors=vectors, namespace=namespace)

# Query
def query(dense_vector: list[float], sparse_vector:dict, namespace:str, top_k:int=5, filter: dict | None=None,alpha:float=0.7) ->list[dict]:
    # dense_vector값과 곱해주는 함수입니다. v가 dense_vector값 이는 70퍼 할당한다는 의미입니다.
    # sparse_vector는 자동으로 30퍼만할당.
    # dense는 번호가 자동할당됨. 저렇게 나뉘지않고
    scaled_dense = [v * alpha for v in dense_vector]
    scaled_sparse ={
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
    return response.get("matches",[])

# 삭제 로직 document_id값 기준
def delete_by_document_id(documnet_id:str,  namespace:str) ->None:
    get_index().delete(
        # eq pincone에서 equal의 의미 같다는 의미로사용함..
        filter={"documnet_id":{"$eq":documnet_id}},
        namespace=namespace,
    )
