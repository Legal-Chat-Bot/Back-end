from pydantic import BaseModel
from datetime import datetime
from uuid import UUID
from typing import Optional 

# 세션 생성 응답
class ChatSessionResponse(BaseModel):
    session_id: UUID
    title: str
    created_at: datetime
    updated_at: datetime
    last_message_at: Optional[datetime] = None

    class Config:
        from_attributes = True

# 이전 메시지 조회 응답용
class MessageResponse(BaseModel):
    message_id: UUID
    session_id: UUID
    user_id: UUID
    question: str
    answer: Optional[str] = None
    is_legal: bool
    question_at: datetime
    answer_at: Optional[datetime] = None

    class Config:
        from_attributes = True 