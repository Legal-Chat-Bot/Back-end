from pydantic import BaseModel, EmailStr, Field, field_serializer, ConfigDict
from enum import Enum
from typing import List
from datetime import datetime

class SocialType(str, Enum):
    NORMAL = "NORMAL"
    GOOGLE = "GOOGLE"
    KAKAO = "KAKAO"

class UserType(str, Enum):
    USER = "USER"
    ADMIN = "ADMIN"

class UserCreate(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    email: EmailStr
    password: str = Field(min_length=8)
    name: str
    social: SocialType = SocialType.NORMAL
    user_type: UserType = UserType.USER

class Login(BaseModel):
    email: EmailStr
    password: str