"""
chat_service 통합 테스트
- process_chat을 통째로 호출해서 전체 RAG 흐름이 도는지 확인
- DB 연결(SessionLocal) 직접 생성해서 넣음
- session_id는 아무 값이나 사용 (DB에 없으면 이전 대화만 빈 채로 진행)
"""

import asyncio
import uuid

from app.db.db import SessionLocal
from app.services.chat_service import process_chat

TEST_QUESTIONS = [
    "퇴직소득의 소득세액은 원칙적으로 어떻게 산정돼?"
]


async def main():
    # DB 연결 직접 생성
    db = SessionLocal()

    # 테스트용 가짜 user_id, session_id (DB에 없어도 됨)
    fake_user_id = str(uuid.uuid4())
    fake_session_id = str(uuid.uuid4())

    try:
        for q in TEST_QUESTIONS:
            print(f"\n{'='*70}")
            print(f"[질문] {q}")
            print('='*70)
            print("처리 중... (CPU 추론, 1~3분)")

            response = await process_chat(
                db=db,
                user_id=fake_user_id,
                session_id=fake_session_id,
                question=q,
            )

            print(f"\n[답변]\n{response.answer}")
            print(f"\n{'-'*70}")
            print(f"검증 통과(verified) : {response.verified}")
            print(f"경고(warnings)       : {response.warnings}")
            print(f"환각 의심 조문        : {response.unverified_refs}")
            print(f"출처 개수(sources)   : {len(response.sources)}")

            # 출처 미리보기
            for i, s in enumerate(response.sources, 1):
                print(f"  출처{i}: [{s.get('category')}] {s.get('excerpt', '')[:40]}...")

    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(main())