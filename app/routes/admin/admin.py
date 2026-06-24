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
     DocumentResponse
)
from app.db.vector.indexer import delete_document_index

router = APIRouter(prefix="/admin", tags=["Admin"])

@router.get(
    "/sessions",
    response_model=list[ChatSessionResponse],
    summary="관리자 전용 채팅방 조회"
)
def admin_get_sessions(
    db: Session = Depends(get_db)
):
    sessions = db.query(Chat).order_by(Chat.updated_at.desc()).all()

    return sessions

@router.get(
    "/messages",
    response_model=list[MessageResponse]
)
def get_messages(
    db: Session = Depends(get_db),
):
    messages = db.query(Message).order_by(Message.question_at.asc()).all()

    return messages

@router.get(
        "/documents",
        response_model=list[DocumentResponse],
        summary="관리자 전용 문서 조회"
)
def admin_documents(
    db: Session = Depends(get_db)
):
    documents = db.query(Document).order_by(Document.created_at.asc()).all()

    return documents

@router.delete(
    "/{document_id}/document",
    summary="관리자용 문서 삭제"
)
async def admin_delete_document(
    document_id: str,
    db: Session = Depends(get_db)
):
    try:   
        document = db.query(Document).filter(
            Document.document_id == document_id
        ).first()

        if not document:
            raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")
        
        delete = await delete_document_index(document=document, db=db)

        db.delete(delete)
        db.commit()

        return {
            "message": "문서가 삭제되었습니다."
        }

    except Exception as e:  
        print("문서 삭제 실패 : ", e)
    finally:
        db.close()