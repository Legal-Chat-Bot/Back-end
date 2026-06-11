import uuid

from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.db import Base

# chat_session 테이블
class Chat(Base):
    __tablename__ = "chat_sessions"

    session_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False
    ) 

    title = Column(String(100), nullable=False)
    last_message_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=True,
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    # ORM user 객체 관계 연결 설정
    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("Message", back_populates="chat_sessions")
    document = relationship("Document", back_populates="chat_sessions")

# Messages 테이블
class Message(Base):
    __tablename__ = "messages"

    message_id = Column(
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

    question = Column(String(100), nullable=False)

    answer = Column(String(100), nullable=False)

    is_legal = Column(Boolean, nullable=False, default=False)

    question_at = Column(DateTime(timezone=True), nullable=False)

    answer_at = Column(DateTime(timezone=True), nullable=False)

    # ORM user객체 chat객체 관계 연결 설정
    user = relationship("User", back_populates="messages")
    chat_sessions = relationship("Chat", back_populates="messages")
