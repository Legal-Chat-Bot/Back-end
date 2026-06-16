from pydantic import BaseModel

class KakaoLoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    is_new_user: bool