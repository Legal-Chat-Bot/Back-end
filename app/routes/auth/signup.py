from fastapi import APIRouter, Depends, status, HTTPException, Request
from sqlalchemy.orm import Session

from app.services.logger_service import create_log, get_trace_id
from app.db.models.logger import Level
from app.schemas.auth.request import UserCreate
from app.schemas.auth.response import UserResponse
from app.db.models.user import User, SocialType
from app.db.db import get_db
from app.core.security import hash_password, get_current_user
from app.services.kakao_service import kakao_unlink
from app.db.vector.client import delete_all

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
def signup(
    user_data: UserCreate,
    request: Request,
    db: Session = Depends(get_db)
):
    trace_id = get_trace_id(request)

    # email로 가입하는 유저 중 같은 이메일을 쓰는지 체크
    existing_user = db.query(User).filter(User.email == user_data.email).first()

    if existing_user:
        create_log(
            db,
            level=Level.WARN,
            endpoint=request.url.path,
            method=request.method,
            status_code=409,
            error_code="SIGNUP_001",
            message=f"회원가입 실패 - 이미 가입된 이메일: {user_data.email}",
            user_id=existing_user.user_id,
            trace_id=trace_id,
        )
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

    create_log(
        db,
        level=Level.INFO,
        endpoint=request.url.path,
        method=request.method,
        status_code=201,
        message=f"회원가입 성공: {new_user.email}",
        user_id=new_user.user_id,
        trace_id=trace_id,
    )

    return new_user

@router.delete(
    "/user",
    summary="회원 탈퇴",
)
async def delete_user(
    request: Request,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    trace_id = get_trace_id(request)

    if not current_user:
        create_log(
            db,
            level=Level.WARN,
            endpoint=request.url.path,
            method=request.method,
            status_code=401,
            error_code="USER_DELETE_001",
            message="회원 탈퇴 실패 - 인증 정보 없음",
            trace_id=trace_id,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="인증이 필요합니다.",
        )

    user = db.query(User).filter(User.email == current_user.email).first()

    if not user:
        create_log(
            db,
            level=Level.WARN,
            endpoint=request.url.path,
            method=request.method,
            status_code=404,
            error_code="USER_DELETE_002",
            message=f"회원 탈퇴 실패 - 사용자 없음: {current_user.email}",
            trace_id=trace_id,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="사용자를 찾을 수 없습니다.",
        )
    deleted_user_id = user.user_id
    deleted_email = user.email

    
    try:
        # 1. Pinecone 네임스페이스 삭제 (안전하게 문자열 변환 및 예외 처리)
        await delete_all(namespace=str(user.user_id)) 
            
        # 카카오 유저면 unlink 추가
        if user.social == SocialType.KAKAO:
            await kakao_unlink(user.social_id)

        db.delete(user)
        db.commit()

        create_log(
            db,
            level=Level.INFO,
            endpoint=request.url.path,
            method=request.method,
            status_code=200,
            message=f"회원 탈퇴 성공: {deleted_email}",
            user_id=deleted_user_id,
            trace_id=trace_id,
        )

    except Exception as e:
        db.rollback()

        create_log(
            db,
            level=Level.ERROR,
            endpoint=request.url.path,
            method=request.method,
            status_code=500,
            error_code="USER_DELETE_999",
            message=f"회원 탈퇴 실패 - 서버 오류: {str(e)}",
            user_id=deleted_user_id,
            trace_id=trace_id,
        )

        raise HTTPException(
            status_code=500,
            detail="회원 탈퇴 처리 중 오류가 발생했습니다.",
        )

    return {"message": "회원 탈퇴가 완료되었습니다."}