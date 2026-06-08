from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.db.models.user import User as UserModel
from app.db.db import get_db
from app.schemas.auth.request import Login as LoginRequest
from app.schemas.auth.response import Token
from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
)

router = APIRouter(tags=["Auth"])


fake_user = {
    "id": 1,
    "email": "test@example.com",
    "password": hash_password("1234"),
    "username": "testuser",
}


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
    db: Session = Depends(get_db)
):
    user = db.query(UserModel).filter(UserModel.email == login_data.email).first()

    print("User found:", user)  # Debugging line to check if the user is found

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="이메일이 올바르지 않습니다.",
        )
    
    if not verify_password(login_data.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="비밀번호가 올바르지 않습니다.",
        )
    token_data = {
        "sub": user.email,
        "user_id": str(user.user_id),
    }

    access_token = create_access_token(data=token_data)
    refresh_token = create_refresh_token(data=token_data)

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
    )


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