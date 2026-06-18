# ============================================================
#
# 텍스트 추출 완료 후 문서 메타데이터 분석 파이프라인
#
# 처리 순서:
#   extracted_text
#     → Chunker (청킹)
#     → DocumentAnalyzer (로컬 LLM분석 및 로컬 모델사용)
#         ├── category  : 문서 카테고리
#         ├── revised_at: 개정일 (YYYY-MM-DD | None, 정규식으로 추출)
#         └── summary   : 문서 요약 텍스트
#
# 설계 원칙:
#   - 청크 전체를 API에 보내지 않고 대표 청크만 샘플링 (토큰 비용 절감)
#   - 개정일은 정규식 1차 추출 → Claude 2차 보완
#   - 카테고리는 사전 정의된 목록 중 선택 (확장 가능)
#   - 실패 시 빈 값 반환 (파이프라인 전체가 죽지 않도록)
# clean_text 옵션
#   summarize() / summarize_document()는 호출 시점에 문서 유형(법률 문서 여부)을
#   모르는 경우가 많다(카테고리 분류 자체가 이 파이프라인의 결과물이기 때문).
#   따라서 기본값은 True로 두고, chunker._split_by_law_structure()가 조 구조가
#   없는 문서는 자동으로 전체 텍스트를 단일 섹션으로 반환하는 안전장치에 보냄.
#   다만 호출하는 쪽(예: 업로드 API)이 문서 유형을 이미 알고 있다면(예: 회의록으로
#   명시 업로드) clean_text=False를 넘겨 불필요한 조 탐지 비용을 줄일 수 있다.
# ============================================================

# Python에서 타입 힌트를 문자열로 처리하도록 하는 기능
from __future__ import annotations

import re
import json
from dataclasses import dataclass
from typing import Optional

from langchain_community.llms import Ollama
from app.core.config import settings
from app.db.vector.chunker import Chunker, ChunkConfig, Chunk
from app.db.vector.summarize_prompt import ANALYSIS_PROMPT_TEMPLATE

# 상수값.

# 로컬 llm 한번만 로드
_llm = Ollama(base_url=settings.OLLAMA_BASE_URL, model=settings.SUMMARIZE_MODEL, temperature=0.1)


# 요약에 사용할 최대 청크
MAX_SAMPLE_CHUNKS = 10

# 요약에 넣기전 분류용 최대 텍스트 길이 (토큰 폭발 방지)
MAX_ANALYSIS_CHARS = 8000

# 지원 카테고리 목록 (필요 시 확장) 안쓸수도 있음. 미정
SUPPORTED_CATEGORIES = [
    "법령·규정",
    "계약서·협약서",
    "판결문·결정문",
    "행정문서·공문",
    "보고서·연구자료",
    "회의록·의사록",
    "매뉴얼·지침서",
    "재무·회계문서",
    "기술문서",
    "기타",
]

# 개정일 정규식 패턴 (우선순위 순)
_RE_DATES: list[re.Pattern] = [
    # "2024년 3월 15일" / "2024년3월15일"
    re.compile(r'(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일'),
    # "2024. 3. 15." / "2024.3.15"
    re.compile(r'(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.?'),
    # "2024-03-15"
    re.compile(r'(\d{4})-(\d{2})-(\d{2})'),
    # "2024/03/15"
    re.compile(r'(\d{4})/(\d{2})/(\d{2})'),
]

# 개정 관련 키워드 (근처에 날짜가 있을 때 우선 선택)
_REVISION_KEYWORDS = re.compile(
    r'(?:개정|시행|제정|공포|최종\s*수정|last\s*revised|amended|effective)',
    re.IGNORECASE,
)


# 데이터 클래스

@dataclass
class DocumentMeta:
    '''
    문서 요약결과
    '''
    category: str = "기타"
    law_date: Optional[str] = None # "YYYY-MM-DD" or None
    summary: str  = ""
    law_name: str = ""
    chunk_count: int = 0 #청크갯수
    error: Optional[str] = None  #오류 메시지

# 날짜 추출 헬퍼

def _extract_revised_date_regex(text: str) -> Optional[str]:
    '''
    텍스트에서 개정일 후보를 정규식으로 추출

    단계:
       1.개정 키워드 주변 200자에서 날짜 탐색 (우선)
       2. 없을경우 문서 앞 500자에서 날짜 탐색 (2번쨰)
       3. 없으면 None
    '''

    def _to_iso(m: re.Match) -> str:
        y, mo, d = m.group(1), m.group(2).zfill(2), m.group(3).zfill(2)
        return f"{y}-{mo}-{d}"

    # 1. 키워드 주변 탐색
    for kw_match in _REVISION_KEYWORDS.finditer(text):
        window_start = max(0, kw_match.start() -50)
        window_end = min(len(text), kw_match.end() +150)
        window = text[window_start:window_end]

        for pat in _RE_DATES:
            m = pat.search(window)
            if m:
                return _to_iso(m)
    

    #2. 문서 앞부분 탐색
    head = text[:500]
    for pat in _RE_DATES:
        m = pat.search(head)
        if m:
            return _to_iso(m)
        
    return None #날짜 못찾으면 "" =>type불일치 문제로인해서 None으로 변경

# 샘플 청크 선택 => 토큰(컨텍스트) 절약용으로 사용하기위해서입니다.

def _sample_chunks(chunks: list[Chunk], n:int = MAX_SAMPLE_CHUNKS) -> list[Chunk]:
    '''
    전체 청크에서 대표 청크를 n개 뽑음.
    
    이번에 사용한방법
    앞: 40% / 중간 20% / 뒤 40% 비율로 균등하게 선택
    짧은 문서일 경우 전체를 사용합니다.
    '''

    if len(chunks) <= n:
        return chunks
    
    front_n = max(1, int(n*0.4))
    middle_n = max(1, int(n*0.2))
    back_n = n - front_n - middle_n

    front = chunks[:front_n]
    back = chunks[-back_n:] if back_n > 0 else []

    mid_start = len(chunks) // 2 - middle_n // 2 
    mid_start = max(front_n, min(mid_start, len(chunks) - back_n - middle_n))
    middle = chunks[mid_start: mid_start + middle_n]


    seen = set()
    result: list[Chunk] = []
    for c in front + middle + back:
        if c.chunk_index not in seen:
            seen.add(c.chunk_index)
            result.append(c)
    
    return result


def _build_sample_text(chunks: list[Chunk]) -> str:
    '''
    샘플 청크를 하나의 텍스트로 조립.
    MAX_ANALYSIS_CHARS
    '''
    parts: list[str] =[]
    total = 0

    for chunk in chunks:
        part = f"[청크 {chunk.chunk_index}]\n{chunk.text}"
        if total + len(part) > MAX_ANALYSIS_CHARS:
            break
        parts.append(part)
        total += len(part)
    
    return "\n\n".join(parts)

# 로컬 llm 호출

def _call_llm(prompt: str) ->str:
    '''
    Ollama로 서빙되는 로컬 llm 호출
    
    실패시 RuntimeError 발생.
    '''
    try:
        return _llm.invoke(prompt).strip()
    except Exception as e:
        raise RuntimeError(f"로컬 LLM 호출 실패: {e}")



# 분석 프롬프트 텍스트,카테고리 ,날짜 나누기

def _build_prompt(analysis_text: str) -> str:
    '''
    로컬 llm에게 전달할 프롬프트

    - category는 SUPPORTED_CATEGORIES 목록 중에서만 선택하도록 강제
    - revised_at은 정규식으로만 처리하므로 LLM에는 요청하지 않음
    - JSON 단독 반환을 강제해 파싱 실패율을 낮춤
    '''

    categories_str = "\n".join(f"  - {c}" for c in SUPPORTED_CATEGORIES)
    return (
        ANALYSIS_PROMPT_TEMPLATE
        .replace("{categories}", categories_str)
        .replace("{text}", analysis_text)
    )

#메인 요약기

class DocumentSummarize:
    '''
    추출된 텍스트 받아서 카테고리·개정일·요약을 반환하는 요약기.

    사용법:
        analyzer = DocumentAnalyzer()
        meta = analyzer.analyze(extracted_text)
        print(meta.category, meta.revised_at, meta.summary)
    '''
    
    def __init__(self, chunk_config: ChunkConfig | None = None):
        self._chunker = Chunker(chunk_config)

    def summarize(self, extracted_text: str, clean_text: bool = True) ->DocumentMeta:
        '''
        텍스트 -> DocumentMeta로 반환

        처리 순서:
          1. 청킹
          2. 정규식 날짜 추출
          3. 샘플 청크 선택 + 요약용 텍스트 조립
          4. 로컬 llm 호출
          5. JSON 파싱 + DocumentMeta 조립
          
          추가
            extracted_text: 문서 본문 텍스트
            clean_text: 조(article) 탐지 여부를 chunker에 그대로 전달합니다.
                - True  (기본값) → 법률 구조(조) 탐지 시도. 구조가 없는 문서는
                  chunker 내부 안전장치(_split_by_law_structure)가 자동으로 값을 넣습니다.
                  전체 텍스트를 단일 섹션으로 처리하므로 일반 문서에도 안전함.
                - False → 조 탐지를 명시적으로 스킵합니다. 
                또한 법률문서 확인을 1차 카테고리로 확인 2차 청킹하고 임베딩해서 vectordb랑 유사도 비교 낮으면 다른파일.
        '''
        #strip 띄워쓰기 방지
        if not extracted_text or not extracted_text.strip():
            return DocumentMeta(error="추출돤 텍스트가 없습니다")
        
        #1.청킹
        try:
            chunks = self._chunker.chunk(extracted_text, clean_text=clean_text)
        except Exception as e:
            return DocumentMeta(error=f"청킹 실패: {e}")
        
        if not chunks:
            return DocumentMeta(error="청킹 결과가 없습니다.")
        
        # 법령명 (chunker로 추출한 값, 첫 청크 기준)
        # clean_text=False면 chunker가 law_name을 채우지 않으므로 자연히 빈 문자열
        law_name = chunks[0].law_name if chunks else ""

        # 2.정규식 날짜 추출
        date_hint = _extract_revised_date_regex(extracted_text)

        # 3. 샘플 청크 선택
        sample_chunks = _sample_chunks(chunks)
        summarize_text = _build_sample_text(sample_chunks)

        # 4. 로컬 llm 호출 - category/ summarty만 담당.
        try:
            prompt = _build_prompt(summarize_text)
            raw_json = _call_llm(prompt)
            result = self._parse_response(raw_json)
        except Exception as e:
            # LLM 실패시 정규식 날짜라도 보존하기
            return DocumentMeta(
                law_date= date_hint,
                law_name = law_name,
                chunk_count= len(chunks),
                error = f"로컬 llm 호출 실패: {e}",
            )
        
        result.law_date = date_hint
        result.law_name = law_name
        result.chunk_count = len(chunks)
        return result
    
    #내부 헬퍼
    #정적 메서드를 사용해서  인스턴스를 만들지 않아도 class의 메서드를 바로 실행하는법.
    @staticmethod
    def _parse_response(raw:str) -> DocumentMeta:
        '''
        로컬 llm 응답 JSON을 파싱해 DocumentMeta로 변환해줌.

        방어 전략:
          - JSON 앞뒤 잡음(```json ... ```) 제거
          - category가 SUPPORTED_CATEGORIES 목록 밖이면 "기타"로 폴백
          - 단 기타면 알수없는 파일이니 사용자는 파일 다시 올리게해야함
          - 파싱 자체 실패 시 error 포함 DocumentMeta 반환 (revised_at은 analyze에서 채움)
        '''
        # ```json ... ``` 펜스 제거
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

        try:
            # json파일 정제.
            data =json.loads(cleaned)
        except json.JSONDecodeError as e:
            return DocumentMeta(
                error = f"JSON 파싱 실패: {e} | raw={raw:}"
            )
        
        
        category = data.get("category", "기타")
        summary = data.get("summary", "")

        #카테고리 유효성 검증
        if category not in SUPPORTED_CATEGORIES:
            category ="기타"

        return DocumentMeta(
            category=category,
            summary= summary,
        )



# 파이프라인 편의함수

def summarize_document(extracted_text:str, chunk_config:ChunkConfig | None = None,clean_text: bool = True,) ->DocumentMeta:
    '''
    단일 함수 인터페이스.

    document_pipeline.py 에서 extract_text_from_file() 결과를 바로 넘기면 됨.

    예시:
        from app.db.vector.document_analyzer import analyze_document
        from app.db.vector.document_pipeline import extract_text_from_file

        text = extract_text_from_file(file_bytes, filename)
        meta = analyze_document(text)

        print(meta.category)    # "법령·규정"
        print(meta.revised_at)  # "2024-01-01"
        print(meta.summary)     # "이 법은 ..."
    '''
    return DocumentSummarize(chunk_config).summarize(extracted_text, clean_text=clean_text)


        




