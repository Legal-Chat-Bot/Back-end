from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session
from pydantic import EmailStr, TypeAdapter, ValidationError

from app.db.db import get_db
from app.schemas.auth.request import UserUpdate
from app.schemas.auth.response import UserResponse
from app.core.security import get_current_user

router = APIRouter(prefix="/user", tags=["User"])

# 이메일 형식 어뎁터
email_adapter = TypeAdapter(EmailStr)

@router.get(
    "/userInfo",
    response_model=UserResponse,
    summary="내 정보 조회"
)
def get_user_info(
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="인증이 필요합니다.",
        )

    return current_user
    

@router.patch(
    "/profile",
    response_model=UserResponse,
    summary="프로필 업데이트"
)
def update_user_profile(
    update_data: UserUpdate,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # 이메일 형식 검증
    if update_data.email is not None:
        try:
            email_adapter.validate_python(update_data.email)
        except ValidationError:
            raise HTTPException(
                status_code=401,
                detail="email 형식을 지켜주세요."
            )

        current_user.email = update_data.email

    if update_data.name is not None:
        current_user.name = update_data.name

    db.commit()
    db.refresh(current_user)

    return current_user