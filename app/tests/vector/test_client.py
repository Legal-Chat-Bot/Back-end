"""
단위 테스트: app/db/vector/client.py
- Pinecone SDK 전체를 mock으로 대체 (네트워크 불필요)
- 실제 프로젝트 적용 시 상단 import 블록 교체 후 인라인 복사본 삭제
"""

import pytest
from unittest.mock import MagicMock, patch, call
from types import SimpleNamespace


# ── 테스트 대상 함수 인라인 복사 ─────────────────────────────────────────────
# 실제 프로젝트에서는 아래로 교체:
# from app.db.vector.client import (
#     get_index, upsert, query, delete_by_ids,
#     delete_by_document_id, delete_all,
#     public_namespace, user_namespace,
# )

# ── 싱글턴 상태 (테스트마다 초기화 필요) ─────────────────────────────────────
_state = {"pc": None, "index": None}

def _reset_state():
    _state["pc"] = None
    _state["index"] = None

def get_index(pinecone_cls, settings):
    if _state["pc"] is None:
        _state["pc"] = pinecone_cls(api_key=settings.PINECONE_API_KEY)

    if _state["index"] is None:
        active = [idx.name for idx in _state["pc"].list_indexes()]
        if settings.PINECONE_INDEX_NAME not in active:
            _state["pc"].create_index(
                name=settings.PINECONE_INDEX_NAME,
                dimension=1024,
                metric="dotproduct",
                spec=object(),
            )
        _state["index"] = _state["pc"].Index(settings.PINECONE_INDEX_NAME)

    return _state["index"]

def public_namespace() -> str:
    return "public"

def user_namespace(user_id: str) -> str:
    return f"user_{user_id}"

def upsert(index, vectors: list[dict], namespace: str) -> dict:
    return index.upsert(vectors=vectors, namespace=namespace)

def query(index, dense_vector, sparse_vector, namespace, top_k=5, filter=None, alpha=0.7):
    scaled_dense = [v * alpha for v in dense_vector]
    scaled_sparse = {
        "indices": sparse_vector["indices"],
        "values": [v * (1 - alpha) for v in sparse_vector["values"]],
    }
    response = index.query(
        vector=scaled_dense,
        sparse_vector=scaled_sparse,
        namespace=namespace,
        top_k=top_k,
        filter=filter,
        include_metadata=True,
    )
    if hasattr(response, "matches"):
        return [m.to_dict() for m in response.matches]
    return response.get("matches", [])

def delete_by_ids(index, vector_ids: list[str], namespace: str) -> None:
    if not vector_ids:
        return
    index.delete(ids=vector_ids, namespace=namespace)

def delete_by_document_id(index, document_id: str, namespace: str) -> None:
    index.delete(
        filter={"document_id": {"$eq": document_id}},
        namespace=namespace,
    )

def delete_all_ns(index, namespace: str) -> None:
    index.delete(delete_all=True, namespace=namespace)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_singleton():
    """각 테스트 전후로 싱글턴 초기화"""
    _reset_state()
    yield
    _reset_state()


@pytest.fixture
def mock_index():
    return MagicMock()


@pytest.fixture
def mock_settings():
    s = SimpleNamespace(
        PINECONE_API_KEY="test-key",
        PINECONE_INDEX_NAME="test-index",
        PINECONE_CLOUD="aws",
        PINECONE_REGION="us-east-1",
    )
    return s


@pytest.fixture
def mock_pinecone_cls(mock_index, mock_settings):
    """list_indexes()에 이미 인덱스가 있는 상황 시뮬레이션"""
    pc_instance = MagicMock()
    existing = SimpleNamespace(name=mock_settings.PINECONE_INDEX_NAME)
    pc_instance.list_indexes.return_value = [existing]
    pc_instance.Index.return_value = mock_index

    cls = MagicMock(return_value=pc_instance)
    return cls, pc_instance


# ── namespace 헬퍼 테스트 ─────────────────────────────────────────────────────

class TestNamespaceHelpers:
    def test_public_namespace(self):
        assert public_namespace() == "public"

    def test_user_namespace(self):
        assert user_namespace("abc123") == "user_abc123"

    def test_user_namespace_uuid(self):
        uid = "550e8400-e29b-41d4-a716-446655440000"
        assert user_namespace(uid) == f"user_{uid}"


# ── get_index 싱글턴 테스트 ───────────────────────────────────────────────────

class TestGetIndex:
    def test_returns_index_object(self, mock_pinecone_cls, mock_settings, mock_index):
        cls, _ = mock_pinecone_cls
        result = get_index(cls, mock_settings)
        assert result is mock_index

    def test_singleton_pinecone_not_created_twice(self, mock_pinecone_cls, mock_settings):
        cls, _ = mock_pinecone_cls
        get_index(cls, mock_settings)
        get_index(cls, mock_settings)
        # Pinecone() 생성자는 첫 호출 때만 1번 불려야 함
        assert cls.call_count == 1

    def test_singleton_index_not_created_twice(self, mock_pinecone_cls, mock_settings, mock_index):
        cls, pc_instance = mock_pinecone_cls
        get_index(cls, mock_settings)
        get_index(cls, mock_settings)
        assert pc_instance.Index.call_count == 1

    def test_creates_index_when_not_exists(self, mock_settings, mock_index):
        """인덱스가 없을 때 create_index를 호출하는지 확인"""
        pc_instance = MagicMock()
        pc_instance.list_indexes.return_value = []   # 빈 목록 → 인덱스 없음
        pc_instance.Index.return_value = mock_index
        cls = MagicMock(return_value=pc_instance)

        get_index(cls, mock_settings)

        pc_instance.create_index.assert_called_once()
        call_kwargs = pc_instance.create_index.call_args.kwargs
        assert call_kwargs["name"] == mock_settings.PINECONE_INDEX_NAME

    def test_skips_create_when_index_exists(self, mock_pinecone_cls, mock_settings):
        """인덱스가 이미 있으면 create_index를 호출하지 않음"""
        cls, pc_instance = mock_pinecone_cls
        get_index(cls, mock_settings)
        pc_instance.create_index.assert_not_called()


# ── query alpha 스케일링 테스트 ───────────────────────────────────────────────

class TestQuery:
    def _make_match(self, score=0.9, id="vec-1", metadata=None):
        m = MagicMock()
        m.to_dict.return_value = {
            "id": id, "score": score, "metadata": metadata or {}
        }
        return m

    def test_alpha_scaling_applied(self, mock_index):
        """alpha=0.6 → dense는 ×0.6, sparse values는 ×0.4"""
        response = MagicMock()
        response.matches = [self._make_match()]
        mock_index.query.return_value = response

        dense = [1.0, 2.0]
        sparse = {"indices": [0, 1], "values": [1.0, 1.0]}
        query(mock_index, dense, sparse, "public", alpha=0.6)

        _, kwargs = mock_index.query.call_args
        assert kwargs["vector"] == pytest.approx([0.6, 1.2])
        assert kwargs["sparse_vector"]["values"] == pytest.approx([0.4, 0.4])

    def test_default_alpha_07(self, mock_index):
        response = MagicMock()
        response.matches = [self._make_match()]
        mock_index.query.return_value = response

        dense = [1.0]
        sparse = {"indices": [0], "values": [1.0]}
        query(mock_index, dense, sparse, "public")

        _, kwargs = mock_index.query.call_args
        assert kwargs["vector"] == pytest.approx([0.7])
        assert kwargs["sparse_vector"]["values"] == pytest.approx([0.3])

    def test_returns_list_of_dicts(self, mock_index):
        m1 = self._make_match(id="a")
        m2 = self._make_match(id="b")
        response = MagicMock()
        response.matches = [m1, m2]
        mock_index.query.return_value = response

        result = query(mock_index, [1.0], {"indices": [0], "values": [1.0]}, "public")

        assert isinstance(result, list)
        assert result[0]["id"] == "a"
        assert result[1]["id"] == "b"

    def test_fallback_to_get_when_no_matches_attr(self, mock_index):
        """matches 속성이 없는 dict 응답 → .get() 폴백"""
        mock_index.query.return_value = {"matches": [{"id": "x"}]}

        result = query(mock_index, [1.0], {"indices": [0], "values": [1.0]}, "public")
        assert result == [{"id": "x"}]

    def test_empty_matches(self, mock_index):
        response = MagicMock()
        response.matches = []
        mock_index.query.return_value = response

        result = query(mock_index, [1.0], {"indices": [0], "values": [1.0]}, "public")
        assert result == []

    def test_filter_passed_through(self, mock_index):
        response = MagicMock()
        response.matches = []
        mock_index.query.return_value = response

        f = {"category": {"$eq": "법령·규정"}}
        query(mock_index, [1.0], {"indices": [0], "values": [1.0]}, "public", filter=f)

        _, kwargs = mock_index.query.call_args
        assert kwargs["filter"] == f

    def test_top_k_passed_through(self, mock_index):
        response = MagicMock()
        response.matches = []
        mock_index.query.return_value = response

        query(mock_index, [1.0], {"indices": [0], "values": [1.0]}, "public", top_k=10)

        _, kwargs = mock_index.query.call_args
        assert kwargs["top_k"] == 10

    def test_namespace_passed_through(self, mock_index):
        response = MagicMock()
        response.matches = []
        mock_index.query.return_value = response

        query(mock_index, [1.0], {"indices": [0], "values": [1.0]}, "user_abc")

        _, kwargs = mock_index.query.call_args
        assert kwargs["namespace"] == "user_abc"


# ── delete_by_ids 테스트 ──────────────────────────────────────────────────────

class TestDeleteByIds:
    def test_calls_delete_with_ids(self, mock_index):
        ids = ["id-1", "id-2"]
        delete_by_ids(mock_index, ids, "public")
        mock_index.delete.assert_called_once_with(ids=ids, namespace="public")

    def test_empty_list_skips_delete(self, mock_index):
        """빈 리스트 → Pinecone API 호출 없음 (API 에러 방지)"""
        delete_by_ids(mock_index, [], "public")
        mock_index.delete.assert_not_called()

    def test_single_id(self, mock_index):
        delete_by_ids(mock_index, ["only-one"], "user_abc")
        mock_index.delete.assert_called_once_with(ids=["only-one"], namespace="user_abc")


# ── delete_by_document_id 테스트 ─────────────────────────────────────────────

class TestDeleteByDocumentId:
    def test_calls_delete_with_filter(self, mock_index):
        delete_by_document_id(mock_index, "doc-123", "public")
        mock_index.delete.assert_called_once_with(
            filter={"document_id": {"$eq": "doc-123"}},
            namespace="public",
        )

    def test_namespace_forwarded(self, mock_index):
        delete_by_document_id(mock_index, "doc-xyz", "user_u1")
        _, kwargs = mock_index.delete.call_args
        assert kwargs["namespace"] == "user_u1"


# ── delete_all 테스트 ─────────────────────────────────────────────────────────

class TestDeleteAll:
    def test_calls_delete_all_true(self, mock_index):
        delete_all_ns(mock_index, "user_abc")
        mock_index.delete.assert_called_once_with(delete_all=True, namespace="user_abc")

    def test_namespace_forwarded(self, mock_index):
        delete_all_ns(mock_index, "user_xyz")
        _, kwargs = mock_index.delete.call_args
        assert kwargs["namespace"] == "user_xyz"


# ── upsert 테스트 ─────────────────────────────────────────────────────────────

class TestUpsert:
    def test_calls_index_upsert(self, mock_index):
        vectors = [{"id": "v1", "values": [0.1, 0.2]}]
        upsert(mock_index, vectors, "public")
        mock_index.upsert.assert_called_once_with(vectors=vectors, namespace="public")

    def test_returns_index_upsert_result(self, mock_index):
        mock_index.upsert.return_value = {"upserted_count": 1}
        result = upsert(mock_index, [{"id": "v1", "values": []}], "public")
        assert result == {"upserted_count": 1}
