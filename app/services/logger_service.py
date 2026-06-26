# app/services/logger_service.py

import uuid
from typing import Optional
from fastapi import Request
from sqlalchemy.orm import Session

from app.db.models.logger import Logger, Level


def get_trace_id(request: Optional[Request] = None) -> str:
    if request and hasattr(request.state, "trace_id"):
        return request.state.trace_id
    return uuid.uuid4().hex


def create_log(
    db: Session,
    *,
    level: Level,
    endpoint: str,
    method: str,
    status_code: int,
    message: str,
    user_id=None,
    session_id=None,
    error_code: Optional[str] = None,
    trace_id: Optional[str] = None,
):
    """
    로그 저장 실패가 실제 API 기능을 막으면 안 되기 때문에
    내부에서 예외를 먹고 rollback만 처리함.
    """
    try:
        log = Logger(
            user_id=user_id,
            session_id=session_id,
            trace_id=trace_id or uuid.uuid4().hex,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            level=level,
            error_code=error_code,
            message=message,
        )

        db.add(log)
        db.commit()

    except Exception:
        db.rollback()