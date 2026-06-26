from sqlalchemy import Column, String, ForeignKey, Text, Enum, Integer, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

import uuid
import enum
from app.db.db import Base

class Level(str, enum.Enum):
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"

class Logger(Base):
    __tablename__ = "logger"

    log_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.session_id", ondelete="CASCADE"),
        nullable=True
    ) 
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=True
    )

    trace_id = Column(String(64), nullable=False, index=True)
    endpoint = Column(String(255), nullable=False)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer, nullable=False)

    level = Column(Enum(Level), nullable=False, default=Level.INFO)

    error_code = Column(String(64), nullable=True)
    message = Column(Text, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )

    # ORM user객체 chat객체 관계 연결 설정
    user = relationship("User", back_populates="logger")
    chat_sessions = relationship("Chat", back_populates="logger")