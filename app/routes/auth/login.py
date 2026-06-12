from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.db.models.user import User as UserModel
from app.db.db import get_db
from app.schemas.auth.request import Login as LoginRequest
from app.schemas.auth.response import Token
from app.core.token_store import token_store
from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    security
)

router = APIRouter(tags=["Auth"])

class RefreshRequest(BaseModel):
    refresh_token: str


@router.post(
    "/auth/login",
    response_model=Token,
    response_model_by_alias=True,
    summary="로그인",
)
def login(
    login_data: LoginRequest,
    db: Session = Depends(get_db)
):
    user = db.query(UserModel).filter(UserModel.email == login_data.email).first()

    if not user:
        raise HTTPException(status_code=401, detail="이메일이 올바르지 않습니다.")

    if not verify_password(login_data.password, user.password):
        raise HTTPException(status_code=401, detail="비밀번호가 올바르지 않습니다.")

    user_id = str(user.user_id)

    # ← 중복 로그인 방지
    if token_store.has_active_session(user_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="이미 로그인된 계정입니다. 먼저 로그아웃 해주세요.",
        )

    token_data = {"sub": user.email, "user_id": user_id}
    access_token = create_access_token(data=token_data)
    refresh_token = create_refresh_token(data=token_data)

    # access token의 jti를 세션에 등록
    access_payload = decode_token(access_token)
    token_store.set_active_session(user_id, access_payload["jti"])

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
    )

@router.post(
    "/auth/logout",
    summary="로그아웃",
)
def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
):
    token = credentials.credentials

    try:
        payload = decode_token(token)
    except ValueError:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다.")

    jti = payload.get("jti")
    exp = payload.get("exp")
    user_id = payload.get("user_id")

    if not jti or not exp or not user_id:
        raise HTTPException(status_code=401, detail="토큰 정보가 올바르지 않습니다.")

    # 블랙리스트 등록 + 세션 제거
    token_store.blacklist_token(jti, exp)
    token_store.remove_active_session(user_id)

    return {"message": "로그아웃 되었습니다."}


@router.post(
    "/auth/refresh",
    response_model=Token,
    response_model_by_alias=True,
    summary="Access Token 재발급",
)
def refresh_token(request: RefreshRequest):
    try:
        payload = decode_token(request.refresh_token)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않은 refresh token입니다.",
        )

    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="refresh token이 아닙니다.",
        )

    email = payload.get("sub")
    user_id = payload.get("user_id")

    if not email or not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="토큰 정보가 올바르지 않습니다.",
        )

    new_access_token = create_access_token({
        "sub": email,
        "user_id": user_id,
    })

    return Token(
        access_token=new_access_token,
        refresh_token=request.refresh_token,
        token_type="bearer",
    )