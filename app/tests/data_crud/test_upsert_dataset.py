import sys
import os
import json
import pytest
from unittest.mock import MagicMock, patch, mock_open

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from dataset.upsert_dataset import upsert_dataset

FAKE_DOC = {
    "article":     "형사소송법 제1조",
    "law_name":    "형사소송법",
    "source_text": "이 법은 형사 절차에 관한 기본 사항을 규정함을 목적으로 한다. 충분히 긴 텍스트입니다.",
    "law_date":    "2026-01-01",
    "info":        {},
}
FAKE_JSONL = json.dumps(FAKE_DOC, ensure_ascii=False) + "\n"


def test_pinecone_failure_triggers_rdb_rollback():
    """
    Pinecone upsert 실패 시 RDB 롤백(delete + commit)이 실행되는지 검증.

    ── 해결 ────────────────────────────────────────────────────
    dataset.upsert_dataset.ChunkDataset 을 MagicMock으로 교체한다.
    그러면 ChunkDataset.dataset_vector_id 도 MagicMock이 되어
    == 비교와 db.query() 인자 전달이 모두 예외 없이 통과된다.
    """

    mock_row = MagicMock()
    mock_row.dataset_vector_id = 1

    mock_embedding = MagicMock()
    mock_embedding.dense  = [0.1] * 10
    mock_embedding.sparse = {"indices": [0], "values": [1.0]}

    mock_chunk = MagicMock()
    mock_chunk.text    = FAKE_DOC["source_text"]
    mock_chunk.article = FAKE_DOC["article"]

    mock_chunker_instance = MagicMock()
    mock_chunker_instance.chunk.return_value = [mock_chunk]

    mock_pinecone = MagicMock()
    mock_pinecone.upsert.side_effect = Exception("Pinecone Network Timeout")

    with patch("builtins.open",                                mock_open(read_data=FAKE_JSONL)), \
         patch("dataset.upsert_dataset.pinecone",             mock_pinecone), \
         patch("dataset.upsert_dataset.Chunker",              return_value=mock_chunker_instance), \
         patch("dataset.upsert_dataset.ChunkDataset"),        \
         patch("dataset.upsert_dataset.create_chunk_dataset", return_value=[mock_row]), \
         patch("dataset.upsert_dataset._embed_texts",         return_value=[mock_embedding]), \
         patch("dataset.upsert_dataset.SessionLocal") as mock_sl:

        # mock_sl.return_value = upsert_dataset 내부 db = SessionLocal() 의 실제 반환 객체
        mock_db = mock_sl.return_value

        try:
            upsert_dataset("dataset/legal.jsonl")
        except Exception:
            pass

        assert mock_pinecone.upsert.called, \
            "pinecone.upsert()가 호출되지 않았습니다 — 롤백 경로 자체에 도달하지 못했습니다."

        assert mock_db.query.called, \
            "db.query()가 호출되지 않았습니다 — RDB 롤백 로직이 실행되지 않았습니다."

        assert mock_db.commit.called, \
            "db.commit()이 호출되지 않았습니다 — 롤백 후 커밋이 누락되었습니다."