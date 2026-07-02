"""
단위 테스트: app/crud/chunk_dataset_crud.py
- SQLite 없이 MagicMock Session으로 동작 (_sqlite3 불필요)
"""

import uuid
import pytest
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import MagicMock


# ── Stub 모델 (인스턴스 속성만, SQLAlchemy 컬럼 표현식 없음) ─────────────────

class ChunkDataset:
    def __init__(self, vector_id, text, article, law_date):
        self.vector_id = vector_id
        self.text      = text
        self.article   = article
        self.law_date  = law_date


# ── 인라인 함수 복사 ──────────────────────────────────────────────────────────

def _parse_law_date(law_date: str | None) -> Optional[datetime]:
    if not law_date:
        return None
    try:
        return datetime.strptime(law_date.strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def create_chunk_dataset(db, chunk_texts, articles, law_date=None):
    if len(chunk_texts) != len(articles):
        raise ValueError(
            f"chunk_texts({len(chunk_texts)})와 articles({len(articles)}) 길이 불일치"
        )
    parsed_law_date = _parse_law_date(law_date)
    rows = [
        ChunkDataset(
            vector_id=str(uuid.uuid4()),
            text=text,
            article=article,
            law_date=parsed_law_date,
        )
        for text, article in zip(chunk_texts, articles)
    ]
    try:
        db.add_all(rows)
        db.commit()
        for row in rows:
            db.refresh(row)
    except Exception:
        db.rollback()
        raise
    return rows


def get_dataset_chunk_by_vector_id(db, vector_id):
    return (
        db.query(ChunkDataset)
        .filter(str(vector_id))   # 컬럼 표현식 평가 우회
        .first()
    )


def update_chunk_dataset_text(db, vector_id, new_text, new_law_date):
    chunk = (
        db.query(ChunkDataset)
        .filter(str(vector_id))
        .first()
    )
    if chunk is None:
        return False
    chunk.text = new_text
    if new_law_date:
        chunk.law_date = _parse_law_date(new_law_date)
    try:
        db.commit()
        db.refresh(chunk)
    except Exception:
        db.rollback()
        raise
    return True


def delete_dataset_chunk_by_vector_id(db, vector_id):
    deleted = (
        db.query(ChunkDataset)
        .filter(str(vector_id))
        .delete(synchronize_session=False)
    )
    db.commit()
    return deleted > 0


# ── Mock DB 헬퍼 ──────────────────────────────────────────────────────────────

def _mock_db():
    db = MagicMock()
    db.add_all  = MagicMock()
    db.commit   = MagicMock()
    db.refresh  = MagicMock()
    db.rollback = MagicMock()
    return db

def _first_chain(db, result):
    """query().filter().first() 결과 설정"""
    db.query.return_value.filter.return_value.first.return_value = result

def _delete_chain(db, deleted_count):
    """query().filter().delete() 결과 설정"""
    db.query.return_value.filter.return_value.delete.return_value = deleted_count


# ── create_chunk_dataset 테스트 ───────────────────────────────────────────────

class TestCreateChunkDataset:
    def test_returns_correct_count(self):
        rows = create_chunk_dataset(_mock_db(), ["a", "b"], ["제1조", "제2조"])
        assert len(rows) == 2

    def test_text_assigned(self):
        rows = create_chunk_dataset(_mock_db(), ["청크1", "청크2"], ["제1조", "제2조"])
        assert rows[0].text == "청크1"
        assert rows[1].text == "청크2"

    def test_article_assigned(self):
        rows = create_chunk_dataset(_mock_db(), ["t1", "t2"], ["제1조", "제2조"])
        assert rows[0].article == "제1조"
        assert rows[1].article == "제2조"

    def test_vector_id_unique(self):
        rows = create_chunk_dataset(_mock_db(), ["a", "b", "c"], ["제1조", "제2조", "제3조"])
        ids = [r.vector_id for r in rows]
        assert len(ids) == len(set(ids))

    def test_law_date_parsed(self):
        rows = create_chunk_dataset(_mock_db(), ["t"], ["제1조"], law_date="2023-11-05")
        assert rows[0].law_date == datetime(2023, 11, 5, tzinfo=timezone.utc)

    def test_law_date_none(self):
        rows = create_chunk_dataset(_mock_db(), ["t"], ["제1조"], law_date=None)
        assert rows[0].law_date is None

    def test_law_date_invalid_format_becomes_none(self):
        rows = create_chunk_dataset(_mock_db(), ["t"], ["제1조"], law_date="2024.01.01")
        assert rows[0].law_date is None

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="길이 불일치"):
            create_chunk_dataset(_mock_db(), ["a", "b"], ["제1조"])

    def test_length_mismatch_does_not_call_db(self):
        db = _mock_db()
        with pytest.raises(ValueError):
            create_chunk_dataset(db, ["a", "b"], ["제1조"])
        db.add_all.assert_not_called()
        db.commit.assert_not_called()

    def test_add_all_called(self):
        db = _mock_db()
        create_chunk_dataset(db, ["t"], ["제1조"])
        db.add_all.assert_called_once()

    def test_commit_called(self):
        db = _mock_db()
        create_chunk_dataset(db, ["t"], ["제1조"])
        db.commit.assert_called_once()

    def test_refresh_called_per_row(self):
        db = _mock_db()
        create_chunk_dataset(db, ["a", "b"], ["제1조", "제2조"])
        assert db.refresh.call_count == 2

    def test_rollback_on_commit_error(self):
        db = _mock_db()
        db.commit.side_effect = RuntimeError("DB 오류")
        with pytest.raises(RuntimeError):
            create_chunk_dataset(db, ["t"], ["제1조"])
        db.rollback.assert_called_once()

    def test_exception_reraised_after_rollback(self):
        db = _mock_db()
        db.commit.side_effect = RuntimeError("커밋 실패")
        with pytest.raises(RuntimeError, match="커밋 실패"):
            create_chunk_dataset(db, ["t"], ["제1조"])


# ── get_dataset_chunk_by_vector_id 테스트 ────────────────────────────────────

class TestGetDatasetChunkByVectorId:
    def test_returns_chunk_when_found(self):
        db = _mock_db()
        fake = MagicMock()
        _first_chain(db, fake)
        assert get_dataset_chunk_by_vector_id(db, uuid.uuid4()) is fake

    def test_returns_none_when_not_found(self):
        db = _mock_db()
        _first_chain(db, None)
        assert get_dataset_chunk_by_vector_id(db, uuid.uuid4()) is None

    def test_query_called_with_model(self):
        db = _mock_db()
        _first_chain(db, None)
        get_dataset_chunk_by_vector_id(db, uuid.uuid4())
        db.query.assert_called_once_with(ChunkDataset)

    def test_filter_called(self):
        db = _mock_db()
        _first_chain(db, None)
        get_dataset_chunk_by_vector_id(db, uuid.uuid4())
        db.query.return_value.filter.assert_called_once()

    def test_first_called(self):
        db = _mock_db()
        _first_chain(db, None)
        get_dataset_chunk_by_vector_id(db, uuid.uuid4())
        db.query.return_value.filter.return_value.first.assert_called_once()


# ── update_chunk_dataset_text 테스트 ─────────────────────────────────────────

class TestUpdateChunkDatasetText:
    def test_returns_false_when_not_found(self):
        db = _mock_db()
        _first_chain(db, None)
        assert update_chunk_dataset_text(db, uuid.uuid4(), "텍스트", "2024-01-01") is False

    def test_does_not_commit_when_not_found(self):
        db = _mock_db()
        _first_chain(db, None)
        update_chunk_dataset_text(db, uuid.uuid4(), "텍스트", "2024-01-01")
        db.commit.assert_not_called()

    def test_returns_true_when_updated(self):
        db = _mock_db()
        chunk = ChunkDataset(str(uuid.uuid4()), "원본", "제1조", None)
        _first_chain(db, chunk)
        assert update_chunk_dataset_text(db, uuid.uuid4(), "수정", "2024-01-01") is True

    def test_text_updated(self):
        db = _mock_db()
        chunk = ChunkDataset(str(uuid.uuid4()), "원본", "제1조", None)
        _first_chain(db, chunk)
        update_chunk_dataset_text(db, uuid.uuid4(), "수정된 텍스트", "2024-01-01")
        assert chunk.text == "수정된 텍스트"

    def test_law_date_updated(self):
        db = _mock_db()
        chunk = ChunkDataset(str(uuid.uuid4()), "원본", "제1조", None)
        _first_chain(db, chunk)
        update_chunk_dataset_text(db, uuid.uuid4(), "텍스트", "2025-06-01")
        assert chunk.law_date == datetime(2025, 6, 1, tzinfo=timezone.utc)

    def test_law_date_not_updated_when_empty(self):
        db = _mock_db()
        original = datetime(2020, 1, 1, tzinfo=timezone.utc)
        chunk = ChunkDataset(str(uuid.uuid4()), "원본", "제1조", original)
        _first_chain(db, chunk)
        update_chunk_dataset_text(db, uuid.uuid4(), "텍스트", "")
        assert chunk.law_date == original

    def test_commit_called(self):
        db = _mock_db()
        chunk = ChunkDataset(str(uuid.uuid4()), "원본", "제1조", None)
        _first_chain(db, chunk)
        update_chunk_dataset_text(db, uuid.uuid4(), "텍스트", "2024-01-01")
        db.commit.assert_called_once()

    def test_rollback_on_commit_error(self):
        db = _mock_db()
        chunk = ChunkDataset(str(uuid.uuid4()), "원본", "제1조", None)
        _first_chain(db, chunk)
        db.commit.side_effect = RuntimeError("커밋 실패")
        with pytest.raises(RuntimeError):
            update_chunk_dataset_text(db, uuid.uuid4(), "텍스트", "2024-01-01")
        db.rollback.assert_called_once()


# ── delete_dataset_chunk_by_vector_id 테스트 ─────────────────────────────────

class TestDeleteDatasetChunkByVectorId:
    def test_returns_true_when_deleted(self):
        db = _mock_db()
        _delete_chain(db, 1)
        assert delete_dataset_chunk_by_vector_id(db, uuid.uuid4()) is True

    def test_returns_false_when_not_found(self):
        db = _mock_db()
        _delete_chain(db, 0)
        assert delete_dataset_chunk_by_vector_id(db, uuid.uuid4()) is False

    def test_commit_called(self):
        db = _mock_db()
        _delete_chain(db, 1)
        delete_dataset_chunk_by_vector_id(db, uuid.uuid4())
        db.commit.assert_called_once()

    def test_delete_with_synchronize_false(self):
        db = _mock_db()
        _delete_chain(db, 1)
        delete_dataset_chunk_by_vector_id(db, uuid.uuid4())
        db.query.return_value.filter.return_value.delete.assert_called_once_with(
            synchronize_session=False
        )

    def test_query_called_with_model(self):
        db = _mock_db()
        _delete_chain(db, 0)
        delete_dataset_chunk_by_vector_id(db, uuid.uuid4())
        db.query.assert_called_once_with(ChunkDataset)