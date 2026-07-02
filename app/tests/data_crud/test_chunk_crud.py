"""
단위 테스트: app/crud/chunk_crud.py
- SQLite 없이 MagicMock Session으로 동작 (_sqlite3 불필요)
- SQLAlchemy 컬럼 표현식(filter 인자)은 query 체인 전체를 mock으로 우회
- 실제 프로젝트 적용 시 import 블록 교체:
    from app.crud.chunk_crud import _parse_law_date, create_chunks_bulk, get_chunks_by_document, delete_chunks_from_rdb
    from app.db.models.chunk import Chunk
"""

import uuid
import pytest
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import MagicMock


# ── Stub 모델 (필터 표현식 없이 인스턴스 속성만 사용) ────────────────────────

class Chunk:
    def __init__(self, vector_id, document_id, chunk_text, article, law_date):
        self.vector_id   = vector_id
        self.document_id = document_id
        self.chunk_text  = chunk_text
        self.article     = article
        self.law_date    = law_date
        self.created_at  = datetime.now(timezone.utc)


# ── 인라인 함수 복사 ──────────────────────────────────────────────────────────

def _parse_law_date(law_date: str | None) -> Optional[datetime]:
    if not law_date:
        return None
    try:
        return datetime.strptime(law_date.strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def create_chunks_bulk(db, chunk_texts, articles, document_id, law_date=None):
    if len(chunk_texts) != len(articles):
        raise ValueError(
            f"chunk_texts({len(chunk_texts)})와 articles({len(articles)}) 길이 불일치"
        )
    parsed_law_date = _parse_law_date(law_date)
    rows = [
        Chunk(
            vector_id=str(uuid.uuid4()),
            document_id=str(document_id),
            chunk_text=text,
            article=article or "",
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


def get_chunks_by_document(db, document_id):
    # 실제 코드는 Chunk.document_id == ... 컬럼 표현식을 사용하므로
    # 테스트에서는 query 체인 전체를 mock으로 대체한다
    return (
        db.query(Chunk)
        .filter(str(document_id))        # 표현식 평가 없이 문자열로 전달
        .order_by("created_at")
        .all()
    )


def delete_chunks_from_rdb(db, document_id):
    deleted = (
        db.query(Chunk)
        .filter(str(document_id))        # 표현식 평가 없이 문자열로 전달
        .delete(synchronize_session=False)
    )
    db.commit()
    return deleted


# ── Mock DB 헬퍼 ──────────────────────────────────────────────────────────────

def _mock_db():
    db = MagicMock()
    db.add_all  = MagicMock()
    db.commit   = MagicMock()
    db.refresh  = MagicMock()
    db.rollback = MagicMock()
    return db


def _query_chain(db, result):
    """query().filter().order_by().all() 체인 결과 설정"""
    (db.query.return_value
       .filter.return_value
       .order_by.return_value
       .all.return_value) = result


def _delete_chain(db, deleted_count):
    """query().filter().delete() 체인 결과 설정"""
    (db.query.return_value
       .filter.return_value
       .delete.return_value) = deleted_count


# ── _parse_law_date 테스트 ────────────────────────────────────────────────────

class TestParseLawDate:
    def test_valid_date(self):
        assert _parse_law_date("2024-01-15") == datetime(2024, 1, 15, tzinfo=timezone.utc)

    def test_none_input(self):
        assert _parse_law_date(None) is None

    def test_empty_string(self):
        assert _parse_law_date("") is None

    def test_whitespace_string(self):
        assert _parse_law_date("   ") is None

    def test_invalid_format_korean(self):
        assert _parse_law_date("2024년 01월 15일") is None

    def test_invalid_format_dot(self):
        assert _parse_law_date("2024.01.15") is None

    def test_invalid_format_slash(self):
        assert _parse_law_date("2024/01/15") is None

    def test_leading_trailing_spaces(self):
        assert _parse_law_date("  2024-03-01  ") == datetime(2024, 3, 1, tzinfo=timezone.utc)

    def test_timezone_is_utc(self):
        assert _parse_law_date("2024-06-01").tzinfo == timezone.utc


# ── create_chunks_bulk 테스트 ─────────────────────────────────────────────────

class TestCreateChunksBulk:
    def test_returns_correct_count(self):
        rows = create_chunks_bulk(_mock_db(), ["a", "b"], ["제1조", "제2조"], uuid.uuid4())
        assert len(rows) == 2

    def test_chunk_text_assigned(self):
        rows = create_chunks_bulk(_mock_db(), ["청크1", "청크2"], ["제1조", "제2조"], uuid.uuid4())
        assert rows[0].chunk_text == "청크1"
        assert rows[1].chunk_text == "청크2"

    def test_article_assigned(self):
        rows = create_chunks_bulk(_mock_db(), ["t1", "t2"], ["제1조", "제2조"], uuid.uuid4())
        assert rows[0].article == "제1조"
        assert rows[1].article == "제2조"

    def test_vector_id_unique(self):
        rows = create_chunks_bulk(_mock_db(), ["a", "b", "c"], ["제1조", "제2조", "제3조"], uuid.uuid4())
        ids = [r.vector_id for r in rows]
        assert len(ids) == len(set(ids))

    def test_document_id_assigned(self):
        doc_id = uuid.uuid4()
        rows = create_chunks_bulk(_mock_db(), ["t"], ["제1조"], doc_id)
        assert rows[0].document_id == str(doc_id)

    def test_law_date_parsed(self):
        rows = create_chunks_bulk(_mock_db(), ["t"], ["제1조"], uuid.uuid4(), law_date="2023-05-10")
        assert rows[0].law_date == datetime(2023, 5, 10, tzinfo=timezone.utc)

    def test_law_date_none(self):
        rows = create_chunks_bulk(_mock_db(), ["t"], ["제1조"], uuid.uuid4(), law_date=None)
        assert rows[0].law_date is None

    def test_law_date_invalid_becomes_none(self):
        rows = create_chunks_bulk(_mock_db(), ["t"], ["제1조"], uuid.uuid4(), law_date="2024.01.01")
        assert rows[0].law_date is None

    def test_article_none_fallback_to_empty(self):
        rows = create_chunks_bulk(_mock_db(), ["t"], [None], uuid.uuid4())
        assert rows[0].article == ""

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="길이 불일치"):
            create_chunks_bulk(_mock_db(), ["a", "b"], ["제1조"], uuid.uuid4())

    def test_length_mismatch_does_not_call_db(self):
        db = _mock_db()
        with pytest.raises(ValueError):
            create_chunks_bulk(db, ["a", "b"], ["제1조"], uuid.uuid4())
        db.add_all.assert_not_called()
        db.commit.assert_not_called()

    def test_add_all_called(self):
        db = _mock_db()
        create_chunks_bulk(db, ["t"], ["제1조"], uuid.uuid4())
        db.add_all.assert_called_once()

    def test_commit_called(self):
        db = _mock_db()
        create_chunks_bulk(db, ["t"], ["제1조"], uuid.uuid4())
        db.commit.assert_called_once()

    def test_refresh_called_per_row(self):
        db = _mock_db()
        create_chunks_bulk(db, ["a", "b"], ["제1조", "제2조"], uuid.uuid4())
        assert db.refresh.call_count == 2

    def test_rollback_on_commit_error(self):
        db = _mock_db()
        db.commit.side_effect = RuntimeError("DB 오류")
        with pytest.raises(RuntimeError):
            create_chunks_bulk(db, ["t"], ["제1조"], uuid.uuid4())
        db.rollback.assert_called_once()

    def test_exception_reraised_after_rollback(self):
        db = _mock_db()
        db.commit.side_effect = RuntimeError("커밋 실패")
        with pytest.raises(RuntimeError, match="커밋 실패"):
            create_chunks_bulk(db, ["t"], ["제1조"], uuid.uuid4())


# ── get_chunks_by_document 테스트 ─────────────────────────────────────────────

class TestGetChunksByDocument:
    def test_returns_chunks(self):
        db = MagicMock()
        fake = [MagicMock(), MagicMock()]
        _query_chain(db, fake)
        result = get_chunks_by_document(db, uuid.uuid4())
        assert result == fake

    def test_returns_empty_list(self):
        db = MagicMock()
        _query_chain(db, [])
        assert get_chunks_by_document(db, uuid.uuid4()) == []

    def test_query_called_with_chunk(self):
        db = MagicMock()
        _query_chain(db, [])
        get_chunks_by_document(db, uuid.uuid4())
        db.query.assert_called_once_with(Chunk)

    def test_filter_called(self):
        db = MagicMock()
        _query_chain(db, [])
        get_chunks_by_document(db, uuid.uuid4())
        db.query.return_value.filter.assert_called_once()

    def test_all_called(self):
        db = MagicMock()
        _query_chain(db, [])
        get_chunks_by_document(db, uuid.uuid4())
        (db.query.return_value
           .filter.return_value
           .order_by.return_value
           .all.assert_called_once())


# ── delete_chunks_from_rdb 테스트 ─────────────────────────────────────────────

class TestDeleteChunksFromRdb:
    def test_returns_deleted_count(self):
        db = MagicMock()
        _delete_chain(db, 3)
        assert delete_chunks_from_rdb(db, uuid.uuid4()) == 3

    def test_returns_zero(self):
        db = MagicMock()
        _delete_chain(db, 0)
        assert delete_chunks_from_rdb(db, uuid.uuid4()) == 0

    def test_commit_called(self):
        db = MagicMock()
        _delete_chain(db, 1)
        delete_chunks_from_rdb(db, uuid.uuid4())
        db.commit.assert_called_once()

    def test_delete_with_synchronize_false(self):
        db = MagicMock()
        _delete_chain(db, 1)
        delete_chunks_from_rdb(db, uuid.uuid4())
        (db.query.return_value
           .filter.return_value
           .delete.assert_called_once_with(synchronize_session=False))

    def test_query_called_with_chunk(self):
        db = MagicMock()
        _delete_chain(db, 0)
        delete_chunks_from_rdb(db, uuid.uuid4())
        db.query.assert_called_once_with(Chunk)