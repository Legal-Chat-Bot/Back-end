from fastapi import APIRouter, Depends, status, HTTPException
from sqlalchemy.orm import Session

from app.schemas.auth.request import UserCreate
from app.schemas.auth.response import UserResponse
from app.db.models.user import User, SocialType
from app.db.db import get_db
from app.core.security import hash_password, get_current_user
from app.services.kakao_service import kakao_unlink
from app.db.vector.client import delete_all,get_index,user_namespace

# app에서 작동하는 것이 아닌 router화로 app과 연동 시켜주기 위한 사전 작업
router = APIRouter(prefix="/auth", tags=["Auth"])

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
    index = get_index()

    # 연결된 db에 유저 데이터를 insert 해준다.
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    usernamespace = user_namespace(new_user.user_id)
    index.create_namespace(name=usernamespace)

    return new_user

@router.delete(
    "/user",
    summary="회원 탈퇴",
)
async def delete_user(
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="인증이 필요합니다.",
        )

    user = db.query(User).filter(User.email == current_user.email).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="사용자를 찾을 수 없습니다.",
        )
    # 1. Pinecone 네임스페이스 삭제 (안전하게 문자열 변환 및 예외 처리)
    try:
        await delete_all(namespace=str(user.user_id)) 
    except Exception as e:
        print(f"[Warning] 회원 탈퇴 중 Pinecone 데이터 삭제 실패 (유저 ID: {user.user_id}): {e}")
    
    # 카카오 유저면 unlink 추가
    if user.social == SocialType.KAKAO:
        await kakao_unlink(user.social_id)

    db.delete(user)
    db.commit()

    return {"message": "회원 탈퇴가 완료되었습니다."}