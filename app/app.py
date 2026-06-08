from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from core.config import settings
from db.db import get_db
from routes.auth.login import router as login_router
from routes.auth.signup import router as signup_router
import os


app = FastAPI(title=settings.PROJECT_NAME)
app.include_router(login_router)
app.include_router(signup_router)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/test")
def read_test(
    db: Session = Depends(get_db)
):
    return {"message": "Hello, World!"}