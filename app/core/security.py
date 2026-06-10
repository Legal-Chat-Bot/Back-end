from fastapi import Depends, status, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from passlib.context import CryptContext

from app.core.config import settings
from app.db.models.user import User
from app.db.db import get_db
from app.core.token_store import token_store
import os
import uuid

SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES
REFRESH_TOKEN_EXPIRE_MINUTES = settings.REFRESH_TOKEN_EXPIRE_MINUTES

# 비밀번호 암호화
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

security = HTTPBearer()

# 비밀번호 hash화
def hash_password(password: str):
    return pwd_context.hash(password)

# hash화 된 비밀번호로 db에 넣기 위한 작업
def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

# token 생성 함수 고유 토큰으로 ID 부여
def create_token(data: dict, expires_delta: timedelta, token_type: str):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + expires_delta

    to_encode.update({
        "exp": expire,
        "type": token_type,
        "jti": str(uuid.uuid4()),
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
    
# 현재 사용자 정보 가져오기 (인증된 사용자)
def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
):
    token = credentials.credentials

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        email = payload.get("sub")
        jti = payload.get("jti")

        if email is None or jti is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="토큰 정보가 올바르지 않습니다.",
            )

        # ← 블랙리스트 체크
        if token_store.is_blacklisted(jti):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="로그아웃된 토큰입니다.",
            )

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않은 토큰입니다.",
        )

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="사용자를 찾을 수 없습니다.",
        )

    return user

# WebSocket 연결에서 토큰 검증 함수
async def verify_ws_token(token: str, db: Session) -> User | None:
    try:
        payload = decode_token(token)
        
        email = payload.get("sub")
        jti = payload.get("jti")

        if not email or not jti:
            return None

        if token_store.is_blacklisted(jti):
            return None

    except ValueError:
        return None

    user = db.query(User).filter(User.email == email).first()
    return user