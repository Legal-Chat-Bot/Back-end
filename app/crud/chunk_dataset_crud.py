from sqlalchemy.orm import Session
import uuid

from app.db.models.chunk_dataset import ChunkDataset
from app.crud.chunk_crud import _parse_law_date


def create_chunk_dataset(
    db: Session,
    chunk_texts: list[str],
    articles: list[str],
    law_date: str | None = None,
) -> list[ChunkDataset]:
    if len(chunk_texts) != len(articles):
        raise ValueError(
            f"chunk_texts({len(chunk_texts)})와 articles({len(articles)}) 길이 불일치"
    )

    parsed_law_date = _parse_law_date(law_date)

    rows = [
        ChunkDataset(                              
            vector_id=uuid.uuid4(),        
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

def get_dataset_chunk_by_vector_id(
    db: Session,
    vector_id: uuid.UUID,
) -> ChunkDataset | None:
    '''Pinecone 유사도 검색 결과 vector_id로 텍스트 조회'''
    return db.query(ChunkDataset).filter(
        ChunkDataset.dataset_vector_id == vector_id
    ).first()

def update_chunk_dataset_text(
    db:Session,
    vector_id: uuid.UUID,
    new_text: str,
    new_law_date: str
) -> bool:
    """
    공용 벡터 갱신 시 RDB의 chunk_text / law_date를 업데이트.
    vector_id는 Pinecone pub_id와 동일하므로 직접 매핑 가능.

    반환: 갱신 성공 여부 (행이 없으면 False)
    """
    chunk = db.query(ChunkDataset).filter(ChunkDataset.vector_id == vector_id).first()
    if chunk is None:
        return False
    chunk.chunk_text = new_text
    if new_law_date:
        chunk.law_date = _parse_law_date(new_law_date)
    try:
        db.commit()
        db.refresh(chunk)
    except Exception:
        db.rollback()
        raise
    return True

def delete_dataset_chunk_by_vector_id(
    db:Session,
    vector_id: uuid.UUID,
) -> bool:
    # 일치하는 vector_id 조회해서 삭제.
    deleted = db.query(ChunkDataset).filter(
        ChunkDataset.dataset_vector_id == vector_id
    ).delete(synchronize_session=False)
    db.commit()
    return deleted > 0

