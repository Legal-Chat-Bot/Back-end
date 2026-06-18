#RDB CRUD
#청킹 적재순서
# 1.청킹
# 2.임베딩 생성.
# 3. 벡터 DB 유사도 검사 (신규 여부 판단)
# 4. RDB 'chunks' 테이블에 먼저 삽입 → DB가 vector_id(PK) 생성
# 5. RDB에서 반환된 vector_id를 Pinecone 벡터의 ID로 사용하여 upsert 로 함.

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from app.db.models.chunk import Chunk

#데이터셋은 일단 law_date를  2024-01-01 형식이지만 유저가 올린파일은 없을수가 있어서.
#None으로 처리 ""로 처리하면 문제생겨서 아래와 같은 문제가생김.
#저장 실패 (Invalid datetime format)
#ORM 에러 발생
#자동 변환 실패
def _parse_law_date(law_date: str | None) -> Optional[datetime]:
    """
    "2024-01-01" 같은 ISO 날짜 문자열을 DateTime 컬럼에 넣을 수 있는
    datetime 객체로 변환. 빈 문자열/None/형식 불일치는 모두 None 처리.
    """
    if not law_date:
        return None
    try:
        return datetime.strptime(law_date.strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    
def create_chunks_bulk(
        db:Session,
        chunk_texts: list[str],
        articles: list[str],
        document_id: uuid.UUID,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
        law_date: str | None = None,
) -> list[Chunk]:
    # 청크당 articles는 1개들어가니 동일해야함.
    if len(chunk_texts) != len(articles):
        raise ValueError(
            f"chunk_texts({len(chunk_texts)})와 articles({len(articles)}) 길이 불일치"
    )
    
    parsed_law_date = _parse_law_date(law_date)
    
    rows = [
        Chunk(
            vector_id=uuid.uuid4(),
            document_id=document_id,
            session_id=session_id,
            user_id=user_id,
            chunk_text=text,
            article=article or "",
            law_date=parsed_law_date,
        )
        #rows = []
        #for text,article in zip(chunk_texts,articles):
        #    rows.append(
        #        Chunk(
        #            vector_id=uuid.uuid4(),
        #            chunk_text=text,
        #            article= article
        #            ...
        #        )
        #    )  for text, article in zip(chunk_texts, articles)
        for text, article in zip(chunk_texts, articles)
    ]
    try:
        db.add_all(rows)
        db.commit()
        for row in rows:
            db.refresh(row)  # DB 기본값(created_at 등)을 인스턴스에 반영
    except Exception:
        #문제 발생하면 db롤백시킴.
        db.rollback()
        raise
    
    return rows

def get_chunks_by_document(db: Session, document_id:uuid.UUID) -> list[Chunk]:
    '''
    특정 문서에 속한 청크 전체 조회 기능 (디버깅겸 검증하기위함)
    '''
    return(
        db.query(Chunk)
        .filter(Chunk.document_id == document_id)
        .order_by(Chunk.created_at.asc()) #처음 들어간것부터확인하기위함. asc
        .all()
    )

#"RDB 전용" vectordb삭제 부분이랑 구분.
def delete_chunks_from_rdb(db: Session, document_id: uuid.UUID) -> int:
    '''
    문서 재인덱싱 or 삭제시 RDB의 기존의 청크를 정리함.
    주의점:반드시 Pinecone 쪽 delete_by_document_id()와 함께 호출해서 양쪽을 맞춰야 한다.
    '''
    deleted =(
        db.query(Chunk)
        .filter(Chunk.document_id== document_id)
        .delete(synchronize_session=False)
    )
    db.commit()
    return deleted

