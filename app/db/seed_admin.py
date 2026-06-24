import os
from sqlalchemy.orm import Session

from app.db.db import SessionLocal
from app.db.models.user import User, UserType
from app.core.security import hash_password
from app.core.config import settings

ADMIN_EMAIL = settings.ADMIN_EMAIL
ADMIN_PASSWORD = settings.ADMIN_PASSWORD
ADMIN_NAME = settings.ADMIN_EMAIL


def create_admin():
    db: Session = SessionLocal()

    try:
        existing_admin = db.query(User).filter(User.email == ADMIN_EMAIL).first()

        if existing_admin:
            print("이미 admin 계정이 존재합니다.")
            return

        admin_user = User(
            email=ADMIN_EMAIL,
            password=hash_password(ADMIN_PASSWORD),
            name=ADMIN_NAME,
            user_type=UserType.ADMIN,
            is_activity=True,
        )

        db.add(admin_user)
        db.commit()

        print("admin 계정 생성 완료")

    except Exception as e:
        db.rollback()
        print(f"admin 계정 생성 실패: {e}")
        raise

    finally:
        db.close()


if __name__ == "__main__":
    create_admin()