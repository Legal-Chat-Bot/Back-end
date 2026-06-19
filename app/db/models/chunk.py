import uuid

from sqlalchemy import Column, String, Text, DateTime,ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.db import Base

class Chunk(Base):
    '''
    청킹정보 테이블
    Pincone에 올라간 벡터 1개당 1개의 행 생성.
    '''
    __tablename__ = "chunks"

    #Pincone 백터 id와 동일해야함.
    vector_id = Column(
        UUID(as_uuid=True),
        primary_key =True,
        default=uuid.uuid4,
        index=True,
    )

    document_id = Column(UUID(as_uuid=True),
                         ForeignKey("document.document_id",ondelete="CASCADE"),
                         nullable=False)

    chunk_text = Column(Text, nullable=False)
    
    # 조(article) 메타데이터
    # indexer.py에서 "법령명 + 조번호" 형태로 조립해서 저장
    #   예) "개인정보 보호법 제12조의2"
    #       "부정청탁 및 금품등 수수의 금지에 관한 법률 제3조"  → 최대 약 40~50자
    # 법령명(최대 ~40자) + 공백 + 조번호(최대 ~9자) = 약 50자정도 되나 더 길수도있음.
    # 여유 있게 String(100)으로 설정하였습니다.
    # 유저가 법률구조가 없는 문서인데 법관련데이터 올릴수있으니 nullable=True, default=""로설정
    article =Column(String(100), nullable=False,default="") # 유저 파일에서 article이 없을수도있음.

    #법령 시행일 / 개정일,
    law_date = Column(DateTime(timezone=True),nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now()
    )

    document = relationship("Document", back_populates="chunks")


# 디버깅용으로 연동후 확인할떄 바로 값이 보이도록하는것. print(chunk)했을떄
# <app.models.chunk.Chunk object at 0x108fa3d90>이런식인게
# <Chunk vector_id=123 document_id='DOC001'> 요런식으로 확인이가능
    def __repr__(self) -> str:
        return (
            f"<Chunk vector_id={self.vector_id} "
            f"document_id={self.document_id}>"
        )


