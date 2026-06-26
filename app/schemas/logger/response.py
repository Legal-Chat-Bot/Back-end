from pydantic import BaseModel, EmailStr, Field, ConfigDict
from uuid import UUID
from typing import Generic, Optional, TypeVar
from datetime import datetime
from app.schemas.auth.request import SocialType, UserType
from app.schemas.constants import StatusCode

class LoggerResponse(BaseModel):
    log_id: UUID
    session_id: UUID | None
    user_id: UUID | None
    trace_id: str | UUID | None
    endpoint: str
    method: str
    status_code: int
    level: str
    error_code: str | None
    message: str
    created_at: datetime 
    
    class Config:
        from_attributes = True