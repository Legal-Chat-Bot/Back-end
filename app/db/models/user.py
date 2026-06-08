import uuid

from sqlalchemy import Column, String, DateTime, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.db.db import Base


class User(Base):
    __tablename__ = "users"

    user_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )

    email = Column(String(100), unique=True, nullable=False, index=True)
    password = Column(String(100), nullable=False)
    name = Column(String(100), nullable=False)

    social = Column(String(20), nullable=False, default="LOCAL")
    social_id = Column(
        UUID(as_uuid=True),
        default=uuid.uuid4,
        nullable=False,
    )

    is_activity = Column(Boolean, nullable=False, default=True)
    user_type = Column(String(20), nullable=False, default="NORMAL")

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
    deleted_at = Column(DateTime(timezone=True), nullable=True)