import httpx
from fastapi import HTTPException, status

from app.core.config import settings

# 카카오 로그인 페이지 URL (사용자를 이 URL로 이동시킴)
KAKAO_AUTH_URL = "https://kauth.kakao.com/oauth/authorize"

# 인가 코드 → 액세스 토큰 발급 URL
KAKAO_TOKEN_URL = "https://kauth.kakao.com/oauth/token"

# 액세스 토큰 → 사용자 정보 조회 URL
KAKAO_USER_INFO_URL = "https://kapi.kakao.com/v2/user/me"


def get_kakao_login_url() -> str:
    """
    카카오 로그인 페이지 URL을 생성해서 반환
    프론트에서 이 URL로 사용자를 이동시키면 카카오 로그인 화면이 뜸

    파라미터 설명:
    - response_type=code : 인가 코드 방식 사용 (OAuth 2.0 표준)
    - client_id          : 우리 앱의 카카오 REST API 키
    - redirect_uri       : 로그인 완료 후 카카오가 인가 코드를 보낼 주소
    """
    return (
        f"{KAKAO_AUTH_URL}"
        f"?response_type=code"
        f"&client_id={settings.KAKAO_REST_API_KEY}"
        f"&redirect_uri={settings.KAKAO_REDIRECT_URI}"
    )


async def get_kakao_access_token(code: str) -> str:
    """
    카카오로부터 받은 인가 코드로 카카오 액세스 토큰을 발급받음

    인가 코드는 1회용이며 짧은 유효시간을 가짐
    액세스 토큰은 카카오 API를 호출할 때 사용

    Args:
        code: 카카오가 redirect_uri로 보내준 인가 코드

    Returns:
        kakao_access_token: 카카오 액세스 토큰 (사용자 정보 조회에 사용)
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            KAKAO_TOKEN_URL,
            data={
                "grant_type": "authorization_code",  # 인가 코드 방식 고정값
                "client_id": settings.KAKAO_REST_API_KEY,      # 카카오 REST API 키
                "redirect_uri": settings.KAKAO_REDIRECT_URI,   # 인가 코드 요청 시 사용한 redirect_uri와 동일해야 함
                "code": code,                          # 카카오로부터 받은 인가 코드
                # client_secret은 카카오 Developers에서 활성화한 경우에만 포함
                **({"client_secret": settings.KAKAO_CLIENT_SECRET} if settings.KAKAO_CLIENT_SECRET else {}),
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    # 토큰 발급 실패 처리
    if response.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"카카오 토큰 발급 실패: {response.text}",
        )

    token_data = response.json()

    # access_token이 응답에 없는 경우 처리
    if "access_token" not in token_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="카카오 액세스 토큰이 응답에 없습니다.",
        )

    return token_data["access_token"]


async def get_kakao_user_info(access_token: str) -> dict:
    """
    카카오 액세스 토큰으로 사용자 정보를 조회

    카카오 동의항목에서 설정한 항목만 받아올 수 있음
    - 닉네임: kakao_account.profile.nickname
    - 이메일: kakao_account.email (동의 필요)

    Args:
        access_token: 카카오 액세스 토큰

    Returns:
        dict: {
            kakao_id: str   → 카카오 고유 ID (우리 DB의 social_id에 저장)
            email: str|None → 이메일 (미동의 시 None)
            nickname: str   → 닉네임
        }
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            KAKAO_USER_INFO_URL,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
            },
        )

    # 사용자 정보 조회 실패 처리
    if response.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"카카오 사용자 정보 조회 실패: {response.text}",
        )

    data = response.json()
    """
    카카오 응답 구조
    {
      "id": 1234567890,                          ← 카카오 고유 ID
      "kakao_account": {
        "email": "user@kakao.com",               ← 이메일 (동의 시)
        "profile": {
          "nickname": "홍길동",                   ← 닉네임
          "profile_image_url": "https://..."     ← 프로필 이미지
        }
      }
    }
    """
    kakao_account = data.get("kakao_account", {})
    profile = kakao_account.get("profile", {})

    return {
        "kakao_id": str(data["id"]),                        # 카카오 고유 ID → social_id에 저장
        "email": kakao_account.get("email"),                # 미동의 시 None
        "nickname": profile.get("nickname", "카카오유저"),   # 닉네임 없으면 기본값
    }