from pydantic import BaseModel, EmailStr, Field, ConfigDict
from uuid import UUID
from typing import Generic, Optional, TypeVar
from datetime import datetime
from app.schemas.auth.request import SocialType, UserType
from app.schemas.constants import StatusCode

T = TypeVar("T")

class BaseResponse(BaseModel, Generic[T]):
    success: bool
    message: Optional[str]
    data: Optional[T]
    code: Optional[int]

class Token(BaseModel):
    access_token: str = Field(serialization_alias="accessToken")
    refresh_token: str = Field(serialization_alias="refreshToken")
    token_type: str = Field(serialization_alias="tokenType")

class UserResponse(BaseModel):
    user_id: UUID
    email: EmailStr
    name: str
    social: SocialType
    user_type: UserType
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)