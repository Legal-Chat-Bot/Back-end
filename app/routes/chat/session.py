import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.db import get_db
from app.db.models.chat import Chat
from app.core.security import get_current_user
from app.db.models.user import User
from app.db.models.chat import Message
from app.db.models.document import Document
from app.schemas.chat.response import(
     ChatSessionResponse,
     MessageResponse,
     DocumentResponse,
     TotalDocumentItem,
     TotalMessageItem,
     TotalItemResponse
)

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

@router.get(
    "/sessions/{session_id}/total",
    response_model=list[TotalItemResponse],
)
def get_session_total(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    messages = db.query(Message).filter(
        Message.session_id == session_id,
        Message.user_id == current_user.user_id,
    ).all()

    documents = db.query(Document).filter(
        Document.session_id == session_id,
        Document.user_id == current_user.user_id,
    ).all()

    total_items = []

    for message in messages:
        total_items.append(
            TotalMessageItem(
                type="message",
                created_at=message.question_at,
                message=MessageResponse.model_validate(message),
            )
        )

    for document in documents:
        total_items.append(
            TotalDocumentItem(
                type="document",
                created_at=document.created_at,
                document=DocumentResponse.model_validate(document),
            )
        )

    total_items.sort(key=lambda item: item.created_at)

    return total_items