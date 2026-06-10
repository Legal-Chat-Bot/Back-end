from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime, timezone
import time

from app.db.db import get_db
from app.db.models.chat import Chat
from app.core.security import verify_ws_token, decode_token
from app.core.token_store import token_store
from app.db.models.chat import Message 

router = APIRouter()

@router.websocket("/ws/chat")
async def websocket_chat(
    websocket: WebSocket,
    token: str = Query(...),
    session_id: Optional[str] = Query(None),  # 있으면 기존, 없으면 새 세션
    db: Session = Depends(get_db),
):
    user = await verify_ws_token(token, db)
    if not user:
        await websocket.close(code=4001)
        return

    await websocket.accept()
    chat_session = None

    # session_id가 있으면 기존 세션 조회
    if session_id:
        chat_session = db.query(Chat).filter(
            Chat.session_id == session_id,
            Chat.user_id == user.user_id,
        ).first()

        if not chat_session:
            await websocket.send_json({
                "type": "error",
                "code": "SESSION_NOT_FOUND",
                "message": "채팅방을 찾을 수 없습니다."
            })
            await websocket.close(code=4004)
            return

    try:
        while True:
            data = await websocket.receive_json()

            # 토큰 재검증
            try:
                payload = decode_token(token)
                jti = payload.get("jti")
                if token_store.is_blacklisted(jti):
                    await websocket.send_json({
                        "type": "error",
                        "code": "TOKEN_BLACKLISTED",
                        "message": "로그아웃된 토큰입니다."
                    })
                    await websocket.close(code=4001)
                    break
            except ValueError:
                await websocket.send_json({
                    "type": "error",
                    "code": "TOKEN_EXPIRED",
                    "message": "토큰이 만료되었습니다."
                })
                await websocket.close(code=4001)
                break

            # 첫 메시지일 때 세션 자동 생성
            if chat_session is None:
                chat_session = Chat(
                    user_id=user.user_id,
                    title=data.get("message")[:20],
                )
                db.add(chat_session)
                db.commit()
                db.refresh(chat_session)

                await websocket.send_json({
                    "type": "session_created",
                    "session_id": str(chat_session.session_id),
                    "title": chat_session.title,
                })

            # ↓ 세션 생성 후 바로 메시지 저장 (기존 세션이든 새 세션이든 동일)
            msg = Message(
                session_id=chat_session.session_id,
                user_id=user.user_id,
                question=data.get("message"),
                question_at=datetime.now(timezone.utc),
            )

            # TODO Ai 호출 후 데이터 응답 답변 코드 여기서 받기
            msg.answer = "AI 응답 데이터"
            msg.answer_at = datetime.now(timezone.utc)
            db.add(msg)
            db.commit()

    except WebSocketDisconnect:
        pass