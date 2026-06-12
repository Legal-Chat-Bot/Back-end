from sqlalchemy import Column, String, ForeignKey, Text, Enum, DateTime, BigInteger, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

import uuid
import enum
from app.db.db import Base

class FileType(str, enum.Enum):
    PPT = "PPT"
    PDF = "PDF"
    HWP = "HWP"
    DOCX = "DOCX"
    XLSX = "XLSX"
    TXT = "TXT"

class Status(str, enum.Enum):
    UPLOADED = "UPLOADED"
    OCR_DONE = "OCR_DONE"
    CHUNKED = "CHUNKED"
    EMBEDDED = "EMBEDDED"
    READY = "READY"
    FAILED = "FAILED"

class Document(Base):
    __tablename__ = "document"

    document_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.session_id", ondelete="CASCADE"),
        nullable=False
    ) 
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False
    )

    file_name = Column(String(100), nullable=False)
    file_ext = Column(Enum(FileType), nullable=False, default=FileType.PDF)
    file_size_bytes = Column(BigInteger, nullable=True)
    storage_url = Column(Text, nullable=False)
    status = Column(Enum(Status), nullable=False, default=Status.READY)
    summary = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    
    # ORM user객체 chat객체 관계 연결 설정
    user = relationship("User", back_populates="document")
    chat_sessions = relationship("Chat", back_populates="document")