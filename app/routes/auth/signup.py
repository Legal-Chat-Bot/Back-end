from fastapi import APIRouter, Depends, status, HTTPException
from sqlalchemy.orm import Session

from app.schemas.auth.request import UserCreate
from app.schemas.auth.response import UserResponse
from app.db.models.user import User
from app.db.db import get_db, engine, Base
from app.core.security import hash_password

# app에서 작동하는 것이 아닌 router화로 app과 연동 시켜주기 위한 사전 작업
router = APIRouter(tags=["Auth"])

Base.metadata.create_all(bind=engine)


"""
회원 가입하기 위한 함수
response 모델로 user 정보를 반환함
"""
@router.post(
    "/signup",
    response_model=UserResponse,
    summary="회원가입",
)
def signup(user_data: UserCreate, db: Session = Depends(get_db)):
    # email로 가입하는 유저 중 같은 이메일을 쓰는지 체크
    existing_user = db.query(User).filter(User.email == user_data.email).first()

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="이미 가입된 이메일입니다.",
        )

    # 가입하지 않았던 이메일이면 해당 유저를 가입하고 new_user 객체에 데이터를 저장
    new_user = User(
        email=user_data.email,
        password=hash_password(user_data.password),
        name=user_data.name,
        social=user_data.social,
        user_type=user_data.user_type,
    )

    # 연결된 db에 유저 데이터를 insert 해준다.
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return new_user