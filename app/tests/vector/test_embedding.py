import sys
import os
from unittest.mock import MagicMock

import numpy as np
import pytest

# 🛠️ 파이썬 패키지 이름 인식 및 경로 충돌 우회 강제 코드
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# 테스트 대상 함수와 해당 모듈을 임포트합니다.
from app.db.vector.embedding import _parse_sparse, embed_texts, embed_query, EmbeddingResult
import app.db.vector.embedding as embedding_mod


# ──────────────────────────────────────────────────────────
# _parse_sparse: dict 포맷 ({"토큰ID": 가중치}) 테스트
# ──────────────────────────────────────────────────────────
class TestParseSparseDictFormat:

    def test_filters_noise_and_normalizes_correctly(self, monkeypatch):
        """
        CASE: BGE-M3 실제 출력 형태인 {"토큰ID": 가중치} 딕셔너리 포맷 검증
        목적: 1. SPARSE_THRESHOLD 이하 노이즈 토큰 제거 확인
              2. 정규화 공식 (각 값 / 최댓값)이 소수점까지 정확히 연산되는지 확인
        """
        monkeypatch.setattr(embedding_mod, "SPARSE_THRESHOLD", 0.15)

        # "101" -> 0.05 (임계값 이하 -> 탈락)
        # "102" -> 0.50 (생존 -> 정규화 후: 0.5 / 2.0 = 0.25)
        # "103" -> 2.00 (생존 및 최댓값 -> 정규화 후: 2.0 / 2.0 = 1.0)
        raw_sparse_mock = {"101": 0.05, "102": 0.5, "103": 2.0}

        result = _parse_sparse(raw_sparse_mock)

        assert 101 not in result["indices"], "임계값 이하인 101번 토큰이 필터링되지 않았습니다."
        assert 102 in result["indices"]
        assert 103 in result["indices"]
        assert isinstance(result["indices"][0], int)

        idx_102 = result["indices"].index(102)
        idx_103 = result["indices"].index(103)

        assert result["values"][idx_103] == 1.0, f"최댓값 정규화 실패: {result['values'][idx_103]}"
        assert result["values"][idx_102] == 0.25, f"비율 정규화 실패: {result['values'][idx_102]}"

    def test_returns_empty_structure_when_all_filtered_out(self, monkeypatch):
        """
        CASE: 모든 토큰 가중치가 임계값 이하라서 몽땅 탈락하는 예외 상황 검증
        목적: values가 비어있을 때 ZeroDivisionError 없이 안전하게 빈 구조를 주는지 확인
        """
        monkeypatch.setattr(embedding_mod, "SPARSE_THRESHOLD", 0.15)

        all_low_data = {"999": 0.01, "888": 0.12}
        result = _parse_sparse(all_low_data)

        assert result["indices"] == []
        assert result["values"] == []

    def test_value_equal_to_threshold_is_excluded(self, monkeypatch):
        """
        CASE: 가중치가 임계값과 '정확히 같은' 경계값(boundary) 검증
        목적: 코드가 `> SPARSE_THRESHOLD` (초과만 통과)를 쓰므로,
              값이 임계값과 동일하면 탈락해야 함. >= 로 잘못 바뀌는 회귀를 방지.
        """
        monkeypatch.setattr(embedding_mod, "SPARSE_THRESHOLD", 0.15)

        boundary_data = {"201": 0.15, "202": 0.150000001}
        result = _parse_sparse(boundary_data)

        assert 201 not in result["indices"], "임계값과 정확히 같은 값은 탈락해야 합니다 (> 비교)."
        assert 202 in result["indices"], "임계값을 초과하는 값은 생존해야 합니다."

    def test_empty_dict_input(self, monkeypatch):
        """CASE: 입력 자체가 빈 dict인 경우"""
        monkeypatch.setattr(embedding_mod, "SPARSE_THRESHOLD", 0.15)

        result = _parse_sparse({})

        assert result["indices"] == []
        assert result["values"] == []

    def test_handles_zero_max_value_without_error(self, monkeypatch):
        """
        CASE: 임계값을 음수로 두어 가중치 0인 토큰도 생존시키는 극단 상황 검증
        목적: max_val == 0 일 때 ZeroDivisionError가 나지 않고 0으로 유지되는지 확인
        """
        monkeypatch.setattr(embedding_mod, "SPARSE_THRESHOLD", -1.0)

        zero_data = {"301": 0.0}
        result = _parse_sparse(zero_data)

        assert result["indices"] == [301]
        assert result["values"] == [0.0]


# ──────────────────────────────────────────────────────────
# _parse_sparse: indices/values 포맷 ({"indices": [...], "values": [...]}) 테스트
# ──────────────────────────────────────────────────────────
class TestParseSparseIndicesValuesFormat:

    def test_indices_values_format_parses_correctly(self, monkeypatch):
        """
        CASE: {"indices": [...], "values": [...]} 형태 입력 검증
        목적: 첫 번째 분기(if "indices" in raw_sparse and "values" in raw_sparse)가
              dict 포맷과 동일하게 필터링 + 정규화되는지 확인
        """
        monkeypatch.setattr(embedding_mod, "SPARSE_THRESHOLD", 0.15)

        raw_sparse_mock = {
            "indices": [101, 102, 103],
            "values": [0.05, 0.5, 2.0],
        }

        result = _parse_sparse(raw_sparse_mock)

        assert 101 not in result["indices"]
        assert 102 in result["indices"]
        assert 103 in result["indices"]

        idx_102 = result["indices"].index(102)
        idx_103 = result["indices"].index(103)
        assert result["values"][idx_103] == 1.0
        assert result["values"][idx_102] == 0.25

    def test_indices_values_empty_lists(self, monkeypatch):
        """CASE: indices/values 키는 있지만 둘 다 빈 리스트인 경우"""
        monkeypatch.setattr(embedding_mod, "SPARSE_THRESHOLD", 0.15)

        raw_sparse_mock = {"indices": [], "values": []}
        result = _parse_sparse(raw_sparse_mock)

        assert result["indices"] == []
        assert result["values"] == []


# ──────────────────────────────────────────────────────────
# _parse_sparse: dict-like 객체 (hasattr indices/values) 테스트
# ──────────────────────────────────────────────────────────
class TestParseSparseDictLikeObject:

    def test_non_dict_object_with_indices_values_attrs_parses_correctly(self, monkeypatch):
        """
        CASE: dict이 아니지만 .indices / .values 속성을 가진 객체 (예: scipy sparse 유사 객체)
        목적: else 분기의 hasattr 처리가 정상 동작하는지 확인
        """
        monkeypatch.setattr(embedding_mod, "SPARSE_THRESHOLD", 0.15)

        class FakeSparseObj:
            indices = [101, 102, 103]
            values = [0.05, 0.5, 2.0]

        result = _parse_sparse(FakeSparseObj())

        assert 101 not in result["indices"]
        assert 102 in result["indices"]
        assert 103 in result["indices"]

        idx_103 = result["indices"].index(103)
        assert result["values"][idx_103] == 1.0

    def test_unsupported_object_without_attrs_returns_empty(self, monkeypatch):
        """
        CASE: dict도 아니고 indices/values 속성도 없는 완전히 미지원 타입
        목적: try/except로 감싸져 있어 예외 없이 빈 구조를 반환하는지 확인
        """
        monkeypatch.setattr(embedding_mod, "SPARSE_THRESHOLD", 0.15)

        result = _parse_sparse(object())

        assert result["indices"] == []
        assert result["values"] == []

    def test_exception_during_attr_access_is_handled_safely(self, monkeypatch):
        """
        CASE: hasattr는 True지만 값을 순회하다 형변환 등에서 예외가 나는 경우
        목적: try/except Exception이 의도대로 잡아 빈 구조를 반환하는지 확인
        """
        monkeypatch.setattr(embedding_mod, "SPARSE_THRESHOLD", 0.15)

        class BrokenSparseObj:
            indices = ["not_an_int_castable_$$$"]
            values = [0.5]

        result = _parse_sparse(BrokenSparseObj())

        assert result["indices"] == []
        assert result["values"] == []


# ──────────────────────────────────────────────────────────
# embed_texts / embed_query: 모델 mock을 이용한 통합 동작 테스트
# ──────────────────────────────────────────────────────────
class TestEmbedTexts:

    @pytest.fixture
    def fake_model(self, monkeypatch):
        """
        get_model()이 반환할 가짜 모델을 주입.
        실제 BGEM3FlagModel을 로드하지 않고 encode() 결과만 흉내낸다.
        """
        monkeypatch.setattr(embedding_mod, "SPARSE_THRESHOLD", 0.15)

        model = MagicMock()

        def fake_encode(texts, batch_size=None, return_dense=True,
                         return_sparse=True, return_colbert_vecs=False):
            n = len(texts)
            return {
                "dense_vecs": np.array([[0.1, 0.2, 0.3] for _ in range(n)]),
                "lexical_weights": [{"101": 0.05, "103": 2.0} for _ in range(n)],
                "colbert_vecs": None,
            }

        model.encode.side_effect = fake_encode
        monkeypatch.setattr(embedding_mod, "get_model", lambda: model)
        return model

    def test_empty_list_input_returns_empty_list(self):
        """CASE: texts=[] 인 경우 모델 호출 없이 즉시 빈 리스트 반환"""
        result = embed_texts([])
        assert result == []

    def test_multiple_texts_each_produce_embedding_result(self, fake_model):
        """
        CASE: 텍스트 2개 입력 시 dense/sparse/text가 올바르게 결합되는지 확인
        """
        texts = ["첫 번째 문장", "두 번째 문장"]
        results = embed_texts(texts)

        assert len(results) == 2
        for i, result in enumerate(results):
            assert isinstance(result, EmbeddingResult)
            assert result.text == texts[i]
            assert result.dense == [0.1, 0.2, 0.3]
            # 노이즈("101": 0.05)는 필터링되고, 103만 남아 정규화 후 1.0
            assert result.sparse["indices"] == [103]
            assert result.sparse["values"] == [1.0]

    def test_model_encode_called_with_correct_options(self, fake_model, monkeypatch):
        """
        CASE: embed_texts가 model.encode를 호출할 때
              return_dense=True, return_sparse=True, return_colbert_vecs=False가
              정확히 전달되는지 확인
        """
        monkeypatch.setattr(embedding_mod, "EMBEDDING_BATCH", 8)

        embed_texts(["테스트 문장"])

        _, kwargs = fake_model.encode.call_args
        assert kwargs["batch_size"] == 8
        assert kwargs["return_dense"] is True
        assert kwargs["return_sparse"] is True
        assert kwargs["return_colbert_vecs"] is False


class TestEmbedQuery:

    @pytest.fixture
    def fake_model(self, monkeypatch):
        monkeypatch.setattr(embedding_mod, "SPARSE_THRESHOLD", 0.15)

        model = MagicMock()

        def fake_encode(texts, **kwargs):
            return {
                "dense_vecs": np.array([[0.4, 0.5, 0.6]]),
                "lexical_weights": [{"500": 0.9}],
                "colbert_vecs": None,
            }

        model.encode.side_effect = fake_encode
        monkeypatch.setattr(embedding_mod, "get_model", lambda: model)
        return model

    def test_single_query_returns_single_embedding_result(self, fake_model):
        """CASE: embed_query는 embed_texts([query])의 첫 번째 결과만 단일로 반환해야 함"""
        result = embed_query("검색 쿼리 문장")

        assert isinstance(result, EmbeddingResult)
        assert result.text == "검색 쿼리 문장"
        assert result.dense == [0.4, 0.5, 0.6]
        assert result.sparse["indices"] == [500]
        assert result.sparse["values"] == [1.0]

    def test_model_called_with_single_item_list(self, fake_model):
        """CASE: embed_query 내부에서 단일 문자열이 리스트로 감싸져 전달되는지 확인"""
        embed_query("쿼리")

        args, _ = fake_model.encode.call_args
        assert args[0] == ["쿼리"]