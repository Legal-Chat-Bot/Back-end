import uuid
import enum

from sqlalchemy import Column, String, DateTime, Boolean, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.db.db import Base

# Python enum 클래스 먼저 정의
class UserType(str, enum.Enum):
    ADMIN = "ADMIN"
    USER = "USER"

class SocialType(str, enum.Enum):
    NORMAL = "NORMAL"
    GOOGLE = "GOOGLE"
    KAKAO = "KAKAO"

class User(Base):
    __tablename__ = "users"

    user_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )

    email = Column(String(100), unique=True, nullable=False, index=True)
    # 소셜 로그인은 password가 없어서
    password = Column(String(100), nullable=False)
    name = Column(String(100), nullable=False)

    social = Column(Enum(SocialType), nullable=False, default=SocialType.NORMAL)
    # social_id: UUID → String으로 변경
    social_id = Column(String(100), nullable=True)  # 카카오 ID는 문자열

    is_activity = Column(Boolean, nullable=False, default=True)
    user_type = Column(Enum(UserType), nullable=False, default=UserType.USER)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    # chat_sessions과 관계
    chat_sessions = relationship("Chat", back_populates="user", cascade="all, delete-orphan")
    messages = relationship("Message", back_populates="user", cascade="all, delete-orphan")
    document = relationship("Document", back_populates="user", cascade="all, delete-orphan")
    chunk = relationship("Chunk", back_populates="user", cascade="all, delete-orphan")
