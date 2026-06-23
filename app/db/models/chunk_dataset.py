import uuid

from sqlalchemy import Column, String, Text, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.db.db import Base

class ChunkDataset(Base):
    '''
    청킹정보 테이블
    Pincone에 올라간 벡터 1개당 1개의 행 생성.
    '''
    __tablename__ = "chunk_dataset"

    #Pincone 백터 id와 동일해야함.
    vector_id = Column(
        UUID(as_uuid=True),
        primary_key =True,
        default=uuid.uuid4,
    )
    chunk_text = Column(Text, nullable=False)

    #법령명+조
    article = Column(String(100), nullable=False, default="")

    #법령 시행일 / 개정일,
    law_date = Column(DateTime(timezone=True),nullable=True)

    #청킹 생성일.
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now()
    )

    #디버깅용.
    def __repr__(self) -> str:
        return (
            f"<DatasetChunk dataset_vector_id={self.dataset_vector_id} "
            f"article={self.dataset_article}>"
        )
