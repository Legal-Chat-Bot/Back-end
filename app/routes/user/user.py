from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session

from app.db.db import get_db
from app.schemas.auth.response import UserResponse
from app.core.security import get_current_user

router = APIRouter(tags=["User"])
@router.get("/userInfo", summary="내 정보 조회")
def get_user_info(
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
    response_model=UserResponse,
):
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="인증이 필요합니다.",
        )

    return current_user
    