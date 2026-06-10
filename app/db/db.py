from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

from app.core.config import settings

DATABASE_URL = settings.POSTGRES_SERVER
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Session = SessionLocal

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    Base.metadata.create_all(bind=engine)  # 그 다음 테이블 생성
