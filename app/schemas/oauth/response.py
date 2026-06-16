from pydantic import BaseModel

class KakaoLoginResponse(BaseModel):
    access_token: str
    refresh_token: str   # 추가
    token_type: str = "bearer"
    is_new_user: bool