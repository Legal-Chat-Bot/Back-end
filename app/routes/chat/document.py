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
from app.db.models.chat import Chat, Message
from app.db.models.document import Document, FileType, Status,Category
from app.schemas.chat.response import DocumentResponse
#vector
from app.db.vector.document_pipeline import extract_text_from_file
from app.db.vector.document_summarize import summarize_document,LAW_CATEGORIES,LAW_UNSTRUCTURED
from app.db.vector.indexer import index_document, delete_document_index

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
        response_model=list[DocumentResponse],
        summary="채팅방 문서 조회"
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

@router.get(
        "/documents",
        response_model=list[DocumentResponse],
        summary="문서 조회"
)
def user_documents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    documents = db.query(Document).filter(
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
        db.commit()
        db.refresh(chat_session)
        

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

    # 텍스트 추출 → 요약
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    
    # 텍스트 추출
    extracted_text = ""
    try:
        extracted_text = extract_text_from_file(file_bytes, original_filename)
    except Exception as e:
        print("텍스트 추출 실패:", e)

        if os.path.exists(file_path):
            os.remove(file_path)

        message = str(e)
        if "UNSUPPORTED_HWPML" in message or "HWPML 2.1" in message or "HWP 3.0" in message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="HWP 3.0 이하 포맷은 지원하지 않습니다.\nHWPX 형식으로 변환 후 다시 업로드해주세요.",
            )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="문서 텍스트 추출에 실패했습니다.",
        )

    meta = None
    summary = ""
    if extracted_text:
        try:
            meta = summarize_document(extracted_text)
            summary = meta.summary
            print("=== 요약 결과 ===")
            print("category:", meta.category)
            print("law_date:", meta.law_date)
            print("law_name:", meta.law_name)
            print("summary:", meta.summary[:100])  # 너무 길면 100자만
        except Exception as e:
            print("요약 실패:", e)     
            summary = ""      

    # 법률 문서 아니면 파일 삭제 후 에러 반환
    if meta is None or meta.category not in (LAW_CATEGORIES | LAW_UNSTRUCTURED):
        os.remove(file_path)  # 저장된 파일도 정리
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="법률 관련 문서만 업로드 가능합니다.",
    )

    # 추후 요약LLM도 생성 후 summary에 적용 시키기
    # Websocket 연동도 같이 진행
    document = Document(
        user_id=current_user.user_id,
        session_id=session_id,
        file_name=original_filename,
        file_ext=file_type,
        file_size_bytes=file_size,
        storage_url=file_path,
        status=Status.CHUNKED,
        category=Category(meta.category),
        summary=summary,
    )

    db.add(document)
    db.commit()
    db.refresh(document)

    try:
        # 청킹테이블에 적재.
        await index_document(
        text=extracted_text,
        document_id=document.document_id,
        user_id=current_user.user_id,
        db=db,
        category=meta.category,
        law_name=meta.law_name or "",
        law_date=meta.law_date or "",
        pre_chunked=meta.chunks,   # ← 청킹 재사용 (두 번 청킹 방지)
        is_public=False,
       )
        document.status = Status.EMBEDDED
        db.commit()
        db.refresh(document)
    except Exception as e:
        print("인덱싱 실패:", e)
        document.status = Status.FAILED
        db.commit()
    finally:
        db.close()
    
    return document

@router.delete(
    "/{document_id}/document",
    summary="사용자용 문서 삭제"
)
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="인증이 필요합니다.",
        )
    
    user = db.query(User).filter(User.email == current_user.email).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="사용자를 찾을 수 없습니다.",
        )
    
    document = db.query(Document).filter(
        User.user_id == current_user.user_id,
        Document.document_id == document_id
    ).first()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="문서를 찾을 수 없습니다."
        )

    try:
        # 문서가 속한 채팅방 ID를 먼저 저장
        session_id = document.session_id

        # 벡터 DB / 인덱스 삭제
        delete_target = await delete_document_index(
            document=document,
            db=db
        )

        # 문서 삭제
        db.delete(delete_target)

        # commit 전 현재 트랜잭션에 삭제 반영
        db.flush()

        # 같은 채팅방에 남은 문서 개수 확인
        document_count = db.query(Document).filter(
            Document.session_id == session_id
        ).count()

        # 같은 채팅방에 남은 메시지 개수 확인
        message_count = db.query(Message).filter(
            Message.session_id == session_id
        ).count()

        chat_session_deleted = False

        # 문서도 없고 메시지도 없으면 채팅방 삭제
        if document_count == 0 and message_count == 0:
            chat_session = db.query(Chat).filter(
                Chat.session_id == session_id,
                Chat.user_id == user.user_id
            ).first()

            if chat_session:
                db.delete(chat_session)
                chat_session_deleted = True

        db.commit()

        return {
            "message": "문서가 삭제되었습니다.",
            "chat_session_deleted": chat_session_deleted
        }
    
    except Exception as e:  
        db.rollback()
        print("문서 삭제 실패 : ", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="문서 삭제에 실패했습니다."
        )