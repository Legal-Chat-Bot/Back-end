import os
import shutil
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.db import get_db
from app.core.config import settings
from app.core.security import get_current_user
from app.db.models.user import User
from app.db.models.chat import Chat
from app.db.models.document import Document, FileType, Status
from app.schemas.chat.response import DocumentResponse

router = APIRouter(prefix="/chat", tags=["Chat"])

# Enum Type 코드
EXTENSION_TO_FILE_TYPE = {
    ".pdf": FileType.PDF,
    ".hwp": FileType.HWP,
    ".hwpx": FileType.HWP,
    ".docx": FileType.DOCX,
    ".ppt": FileType.PPT,
    ".pptx": FileType.PPT,
    ".xlsx": FileType.XLSX,
    ".txt": FileType.TXT,
}

# 파일 확장자 필터링 코드
def get_file_type(filename: str) -> FileType:
    ext = os.path.splitext(filename)[1].lower()

    file_type = EXTENSION_TO_FILE_TYPE.get(ext)

    if not file_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"지원하지 않는 파일 형식입니다: {ext}",
        )

    return file_type

# 기본 파일명
def get_original_filename(filename: str) -> str:
    return os.path.basename(filename)

# 파일명 uuid로 변경
def make_storage_filename(original_filename: str) -> str:
    ext = os.path.splitext(original_filename)[1].lower()
    return f"{uuid.uuid4().hex}{ext}"

@router.get(
        "/sessions/{session_id}/documents",
        response_model=list[DocumentResponse]
)
def get_documents(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    documents = db.query(Document).filter(
        Document.session_id == session_id,
        Document.user_id == current_user.user_id,
    ).order_by(Document.created_at.asc()).all()

    return documents

@router.post(
    "/sessions/{session_id}/upload",
    response_model = DocumentResponse
)
async def upload_file(
    session_id: str,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="파일명이 없습니다.",
        )

    original_filename = get_original_filename(file.filename)
    file_type = get_file_type(original_filename)

    # 선택된 채팅방 탐색
    chat_session = db.query(Chat).filter(
        Chat.session_id == session_id,
        Chat.user_id == current_user.user_id,
    ).first()

    # 없으면 새로 생성
    if not chat_session:
        chat_session = Chat(
            session_id=session_id,
            user_id=current_user.user_id,
            title=original_filename,  # 파일명을 제목으로
        )
        db.add(chat_session)
        db.flush()  # commit 전에 DB에 반영 (document FK 연결 위해)

    # settings에서 설정한 directory 접근
    upload_dir = settings.UPLOAD_DIR
    # 해당 directory가 없으면 생성
    os.makedirs(upload_dir, exist_ok=True)

    # UUID 난독화 작업한 파일명을 file_path로 저장
    storage_filename = make_storage_filename(original_filename)
    file_path = os.path.join(upload_dir, storage_filename)

    # 난독화한 파일을 설정한 Directory에 복사하여 파일을 저장
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_size = os.path.getsize(file_path)

    # 추후 요약LLM도 생성 후 summary에 적용 시키기
    # Websocket 연동도 같이 진행
    document = Document(
        user_id=current_user.user_id,
        session_id=session_id,
        file_name=original_filename,
        file_ext=file_type,
        file_size_bytes=file_size,
        storage_url=file_path,
        status=Status.UPLOADED,
        summary="",
    )

    db.add(document)
    db.commit()
    db.refresh(document)

    return document