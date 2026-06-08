from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from passlib.context import CryptContext
from app.core.config import settings
import os

SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES
REFRESH_TOKEN_EXPIRE_MINUTES = settings.REFRESH_TOKEN_EXPIRE_MINUTES

# 비밀번호 암호화
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 비밀번호 hash화
def hash_password(password: str):
    return pwd_context.hash(password)

# hash화 된 비밀번호로 db에 넣기 위한 작업
def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

# token 생성 함수
def create_token(data: dict, expires_delta: timedelta, token_type: str):
    to_encode = data.copy()

    expire = datetime.now(timezone.utc) + expires_delta

    to_encode.update({
        "exp": expire,
        "type": token_type,
    })

    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# access token 생성 함수
def create_access_token(data: dict) -> str:
    return create_token(
        data=data,
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        token_type="access",
    )

# refresh token 생성 함수
def create_refresh_token(data: dict) -> str:
    return create_token(
        data=data,
        expires_delta=timedelta(minutes=REFRESH_TOKEN_EXPIRE_MINUTES),
        token_type="refresh",
    )

# token 디코딩 함수
def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise ValueError("Invalid token")