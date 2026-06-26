from fastapi import APIRouter, HTTPException, Depends, status, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.db.models.user import User, SocialType
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
from app.services.kakao_service import kakao_logout
from app.services.logger_service import create_log, get_trace_id
from app.db.models.logger import Level


router = APIRouter(prefix="/auth", tags=["Auth"])

class RefreshRequest(BaseModel):
    refresh_token: str


@router.post(
    "/login",
    response_model=Token,
    response_model_by_alias=True,
    summary="로그인",
)
def login(
    login_data: LoginRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    trace_id = get_trace_id(request)

    user = db.query(User).filter(User.email == login_data.email).first()

    if not user:
        create_log(
            db,
            level=Level.WARN,
            endpoint=request.url.path,
            method=request.method,
            status_code=401,
            error_code="AUTH_001",
            message=f"로그인 실패 - 존재하지 않는 이메일: {login_data.email}",
            trace_id=trace_id,
        )
        raise HTTPException(status_code=401, detail="이메일이 올바르지 않습니다.")

    if not verify_password(login_data.password, user.password):
        create_log(
            db,
            level=Level.WARN,
            endpoint=request.url.path,
            method=request.method,
            status_code=401,
            error_code="AUTH_002",
            message=f"로그인 실패 - 비밀번호 불일치: {user.email}",
            user_id=user.user_id,
            trace_id=trace_id,
        )
        raise HTTPException(status_code=401, detail="비밀번호가 올바르지 않습니다.")

    user_id = str(user.user_id)

    # ← 중복 로그인 방지
    if token_store.has_active_session(user_id):
        create_log(
            db,
            level=Level.WARN,
            endpoint=request.url.path,
            method=request.method,
            status_code=409,
            error_code="AUTH_003",
            message=f"로그인 실패 - 이미 로그인된 계정: {user.email}",
            user_id=user.user_id,
            trace_id=trace_id,
        )
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

    create_log(
        db,
        level=Level.INFO,
        endpoint=request.url.path,
        method=request.method,
        status_code=200,
        message=f"로그인 성공: {user.email}",
        user_id=user.user_id,
        trace_id=trace_id,
    )

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
    )

@router.post(
    "/logout",
    summary="로그아웃",
)
async def logout(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
):
    trace_id = get_trace_id(request)

    token = credentials.credentials

    try:
        payload = decode_token(token)
    except ValueError:
        create_log(
            db,
            level=Level.WARN,
            endpoint=request.url.path,
            method=request.method,
            status_code=401,
            error_code="LOGOUT_001",
            message="로그아웃 실패 - 유효하지 않은 토큰",
            trace_id=trace_id,
        )
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다.")

    jti = payload.get("jti")
    exp = payload.get("exp")
    user_id = payload.get("user_id")

    if not jti or not exp or not user_id:
        create_log(
            db,
            level=Level.WARN,
            endpoint=request.url.path,
            method=request.method,
            status_code=401,
            error_code="LOGOUT_002",
            message="로그아웃 실패 - 토큰 정보 누락",
            trace_id=trace_id,
        )
        raise HTTPException(status_code=401, detail="토큰 정보가 올바르지 않습니다.")
    
    # DB에서 유저 조회
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        create_log(
            db,
            level=Level.WARN,
            endpoint=request.url.path,
            method=request.method,
            status_code=404,
            error_code="LOGOUT_003",
            message=f"로그아웃 실패 - 유저 없음: {user_id}",
            trace_id=trace_id,
        )
        raise HTTPException(status_code=404, detail="유저를 찾을 수 없습니다.")
    
    # 카카오 유저면 unlink 추가
    if user.social == SocialType.KAKAO:
        await kakao_logout(user.social_id)

    # 블랙리스트 등록 + 세션 제거
    token_store.blacklist_token(jti, exp)
    token_store.remove_active_session(user_id)

    create_log(
        db,
        level=Level.INFO,
        endpoint=request.url.path,
        method=request.method,
        status_code=200,
        message=f"로그아웃 성공: {user.email}",
        user_id=user.user_id,
        trace_id=trace_id,
    )

    return {"message": "로그아웃 되었습니다."}


@router.post(
    "/refresh",
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