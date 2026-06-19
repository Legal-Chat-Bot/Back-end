from pydantic import BaseModel

class KakaoCallbackRequest(BaseModel):
    """
    프론트에서 백엔드로 전달하는 요청 스키마
    카카오가 redirect_uri로 보내준 인가 코드를 담음
    """
    code: str  # 카카오 인가 코드 (1회용, 짧은 시간 내 사용해야 함)