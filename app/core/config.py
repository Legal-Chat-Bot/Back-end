from pydantic_settings import BaseSettings
from pydantic import Field 
from typing import List



class Settings(BaseSettings):
    PROJECT_NAME: str = "법률 챗봇"

    # 모델설정
    OLLAMA_BASE_URL: str = "http://localhost:11434" #모델 url
    # 역할에 따라 모델 분리
    # str은 헌팅식 python 문법입니다. typescript처럼 타입을 미리 정의를 해주는방식
    CLASSIFIER_MODEL: str = "qwen2.5:3b"    # 문서 분류용
    EMBEDDING_MODEL: str = "bge-m3"         # 임베딩용
    RAG_MODEL: str = "llama3.2:3b"          # RAG 응답용

    # 허용 주소값
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000"]

    # pincone
    # `...`(Ellipsis)은 "기본값은 비운다는 의미지만 env에서 시스템환경변수 같은 값을 최우선으로 가져오는 의미입니다.
    PINECONE_API_KEY: str = Field(default=...)
    PINECONE_INDEX_NAME: str = Field(default=...)
    PINECONE_CLOUD: str = Field(default=...)
    PINECONE_REGION: str = Field(default=...)

    # db세팅.
    DATABASE_URL: str = Field(default=...)


    class Config:
        env_file = "../.env"
        env_file_encoding = "utf-8"


settings = Settings()
