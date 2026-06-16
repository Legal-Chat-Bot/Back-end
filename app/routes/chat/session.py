import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.db import get_db
from app.db.models.chat import Chat
from app.core.security import get_current_user
from app.db.models.user import User
from app.db.models.chat import Message
from app.schemas.chat.response import ChatSessionResponse, MessageResponse

router = APIRouter(prefix="/chat", tags=["Chat"])

# 채팅방 생성
@router.post(
    "/session")
def create_session(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = Chat(
        user_id=current_user.user_id,
        title="새 대화",
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    return {
        "session_id": session.session_id,
        "title": session.title,
        "created_at": session.created_at,
    }

# 채팅방 목록 조회
@router.get(
    "/sessions",
    response_model=list[ChatSessionResponse]
)
def get_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    sessions = db.query(Chat).filter(
        Chat.user_id == current_user.user_id
    ).order_by(Chat.updated_at.desc()).all()

    return sessions

# 채팅방 입장할 때 이전 메시지 불러오기
@router.get(
    "/sessions/{session_id}/messages",
    response_model=list[MessageResponse]
)
def get_messages(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    messages = db.query(Message).filter(
        Message.session_id == session_id,
        Message.user_id == current_user.user_id,
    ).order_by(Message.question_at.asc()).all()

    return messages