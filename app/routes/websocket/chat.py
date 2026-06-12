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

    # 기존 세션으로 입장하는 경우
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

            message_text = (data.get("message") or "").strip()

            if not message_text:
                await websocket.send_json({
                    "type": "error",
                    "code": "EMPTY_MESSAGE",
                    "message": "메시지가 비어 있습니다.",
                })
                continue

            now = datetime.now(timezone.utc)

            # 첫 메시지일 때 세션 자동 생성
            if chat_session is None:
                chat_session = Chat(
                    user_id=user.user_id,
                    title=message_text[:20],
                )

                db.add(chat_session)
                db.commit()
                db.refresh(chat_session)

                await websocket.send_json({
                    "type": "session_created",
                    "session_id": str(chat_session.session_id),
                    "title": chat_session.title,
                })

            # 메시지 저장
            msg = Message(
                session_id=chat_session.session_id,
                user_id=user.user_id,
                question=message_text,
                question_at=datetime.now(timezone.utc),
            )

            # TODO: 여기에 실제 AI 호출 로직 연결
            ai_answer = "AI 응답 데이터"

            msg.answer = ai_answer
            msg.answer_at = datetime.now(timezone.utc)

            # 채팅방 updated_at 갱신
            if hasattr(chat_session, "updated_at"):
                chat_session.updated_at = datetime.now(timezone.utc)

            db.add(msg)
            db.commit()
            db.refresh(msg)

            await websocket.send_json({
                "type": "message",
                "session_id": str(chat_session.session_id),
                "message_id": str(getattr(msg, "message_id", "")),
                "question": msg.question,
                "answer": msg.answer,
                "question_at": msg.question_at.isoformat()
                if msg.question_at
                else None,
                "answer_at": msg.answer_at.isoformat()
                if msg.answer_at
                else None,
            })

    except WebSocketDisconnect:
        pass

    except Exception:
        db.rollback()

        try:
            await websocket.send_json({
                "type": "error",
                "code": "SERVER_ERROR",
                "message": "서버 처리 중 오류가 발생했습니다.",
            })
            await websocket.close(code=1011)
        except Exception:
            pass