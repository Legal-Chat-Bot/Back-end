from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.db import get_db
from app.db.models.user import User, SocialType
from app.schemas.oauth.request import KakaoCallbackRequest
from app.schemas.oauth.response import KakaoLoginResponse
from app.services.kakao_service import (
    get_kakao_login_url,
    get_kakao_access_token,
    get_kakao_user_info,
)
from app.core.security import create_access_token, create_refresh_token  # 기존 JWT 발급 함수

router = APIRouter(prefix="/auth/kakao", tags=["OAUTH"])

@router.get("/login-url")
def kakao_login_url():
    """
    [GET] /auth/kakao/login-url

    프론트에서 카카오 로그인 버튼 클릭 시 호출
    카카오 로그인 페이지 URL을 반환
    
    """
    login_url = get_kakao_login_url()
    return {"login_url": login_url}


@router.post("/callback", response_model=KakaoLoginResponse)
async def kakao_callback(
    body: KakaoCallbackRequest,
    db: Session = Depends(get_db),
):
    """
    [POST] /auth/kakao/callback

    카카오 로그인 완료 후 프론트가 인가 코드를 여기로 전달
    전체 처리 순서:
    1. 인가 코드 → 카카오 액세스 토큰 발급
    2. 카카오 액세스 토큰 → 카카오 사용자 정보 조회
    3. DB 조회 → 기존 회원이면 로그인 / 신규면 자동 회원가입
    4. 우리 서비스 JWT 발급 후 프론트로 반환

    Args:
        body.code: 카카오가 redirect_uri로 보내준 인가 코드

    Returns:
        access_token: 우리 서비스 JWT
        token_type: "bearer"
        is_new_user: 신규 가입 여부 (프론트에서 UI 분기용)
    """

    # ── 인가 코드 → 카카오 액세스 토큰 ──
    kakao_access_token = await get_kakao_access_token(body.code)

    # ── 카카오 액세스 토큰 → 사용자 정보 ──
    kakao_user = await get_kakao_user_info(kakao_access_token)

    kakao_id = kakao_user["kakao_id"]      # 카카오 고유 ID
    email = kakao_user.get("email")         # 이메일 (미동의 시 None)
    nickname = kakao_user.get("nickname", "카카오유저")

    # social == KAKAO 이고 social_id == 카카오 ID 인 유저 조회
    user = (
        db.query(User)
        .filter(
            User.social == SocialType.KAKAO,
            User.social_id == kakao_id,
        )
        .first()
    )

    is_new_user = False  # 신규 가입 여부 플래그

    if not user:
        # 신규 유저 → 자동 회원가입 ──

        # 같은 이메일로 일반 가입(NORMAL)된 계정이 있는지 체크
        # (카카오 이메일 = 기존 일반 가입 이메일 충돌 방지)
        if email:
            existing_user = db.query(User).filter(User.email == email).first()
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="이미 해당 이메일로 가입된 계정이 있습니다. 일반 로그인을 이용해주세요.",
                )

        # 새 유저 생성
        user = User(
            # 이메일 없는 경우(미동의) kakao_id로 임시 이메일 생성
            email=email or f"kakao_{kakao_id}@kakao.local",
            password="",              # 소셜 로그인은 비밀번호 없음
            name=nickname,
            social=SocialType.KAKAO,  # 소셜 타입 KAKAO로 설정
            social_id=kakao_id,       # 카카오 고유 ID 저장
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        is_new_user = True  # 신규 가입 플래그

    # ── 우리 서비스 JWT 발급 ──
    # 기존 일반 로그인과 동일한 방식으로 JWT 발급
    access_token = create_access_token(data={"sub": user.email})
    refresh_token = create_refresh_token(data={"sub": user.email})

    return KakaoLoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        is_new_user=is_new_user,
    )