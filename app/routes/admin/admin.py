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
from app.db.models.logger import Logger, Level

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
        
        # 문서가 속한 채팅 세션 ID 미리 저장
        # 네 모델 필드명이 다르면 여기만 수정
        session_id = document.session_id

        # 벡터 DB / 인덱스 삭제 처리
        # 이 함수가 document 객체를 반환한다고 가정
        delete_target = await delete_document_index(
            document=document,
            db=db
        )

        # 문서 삭제
        db.delete(delete_target)

        # 아직 commit 전이지만, 현재 트랜잭션 안에서 삭제 상태 반영
        db.flush()

        # 같은 세션에 남은 문서 개수 확인
        document_count = db.query(Document).filter(
            Document.session_id == session_id
        ).count()

        # 같은 세션에 남은 메시지 개수 확인
        message_count = db.query(Message).filter(
            Message.session_id == session_id
        ).count()

        chat_session_deleted = False

        # 8. 문서도 없고 메시지도 없으면 채팅방 삭제
        if document_count == 0 and message_count == 0:
            chat_session = db.query(Chat).filter(
                Chat.session_id == session_id
            ).first()

            if chat_session:
                db.delete(chat_session)
                chat_session_deleted = True

        # 9. 최종 커밋
        db.commit()

        return {
            "message": "문서가 삭제되었습니다.",
            "chat_session_deleted": chat_session_deleted
        }

    except Exception as e:  
        print("문서 삭제 실패 : ", e)
    finally:
        db.close()

router.get(
    "/logger/info",
    summary="관리자용 활동 로그 조회"
)
def get_logger_info(
    db: Session = Depends(get_db)
):
    logger = db.query(Logger).filter(
        Logger.level == Level.INFO
    ).order_by(Logger.created_at.desc()).all()

    return logger

router.get(
    "/logger/error",
    summary="관리자용 에러 로그 조회"
)
def get_logger_error(
        db: Session = Depends(get_db)
):
    logger = db.query(Logger).filter(
        Logger.level.in_([Level.WARN, Level.ERROR])
    ).order_by(Logger.created_at.desc()).all()

    return logger