from pydantic import BaseModel, EmailStr, Field
from typing import Generic, Optional, TypeVar
from datetime import datetime
from schemas.auth.request import SocialType, UserType
from schemas.constants import StatusCode

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
    id: int
    email: EmailStr
    username: str
    social: SocialType
    user_type: UserType
    created_at: datetime