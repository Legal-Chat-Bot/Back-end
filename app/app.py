from fastapi import FastAPI, Depends
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.db import get_db, init_db
from app.routes.auth.login import router as login_router
from app.routes.auth.signup import router as signup_router
from app.routes.user.user import router as user_router
from app.routes.websocket.chat import router as websocket_router
from app.routes.chat.session import router as chat_router
from app.routes.chat.document import router as document_router
import os

app = FastAPI(title=settings.PROJECT_NAME)
app.include_router(login_router)
app.include_router(signup_router)
app.include_router(user_router)
app.include_router(chat_router)
app.include_router(document_router)
app.include_router(websocket_router)

init_db()  # 애플리케이션 시작 시 DB 초기화 (테이블 생성 등)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def get():
    return {"message": "Hello, World!"}

@app.get("/test")
def read_test(
    db: Session = Depends(get_db)
):
    return {"message": "Hello, World!"}