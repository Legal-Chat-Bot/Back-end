# 청킹 파이프라인
#
# 문서 → 조 구조 분리(1차) → KSS 문장 분리(2차) → 청크 조립 → 청크 리스트 반환
#
# 개선 사항:
#   - 법률 문서 구조(조) 기반 1차 분리 추가
#   - Chunk에 article(조) 메타데이터 추가
#   - already_cleaned 플래그로 정제 스킵 지원
#   - 텍스트에서 법령명 자동 추출 (extract_law_name)
#   - Chunk에 law_name 메타데이터 추가

from dataclasses import dataclass, field
from typing import Callable
import re
import unicodedata
from kss import Kss



# KSS 모듈 초기화(한번만)
_split_sentences = Kss("split_sentences")
# 유저가 올린 문서에 한문데이터 한글로변환
_hanja2hangul = Kss("hanja2hangul")

# ── 법률 구조 파싱용 정규식 ────────────────────────────────

# 조 경계: "제1조", "제1조의2", "제12조" 등
_RE_ARTICLE = re.compile(
    r'(?=제\s*\d+\s*조(?:의\s*\d+)?(?:\s*[\(（][^\)）]*[\)）])?)',
    re.MULTILINE,
)

# 조 번호 추출용 (텍스트에서 "제N조" 추출)
_RE_ARTICLE_NUM = re.compile(r'제\s*(\d+)\s*조(?:의\s*(\d+))?')

# 법령명 추출용
# "구 X법", "X법", "X에 관한 법률" 등 + 뒤에 "(..." 또는 "제N조" 또는 공백이 오는 패턴
# "구 " 접두사, 괄호 안 연도 표현(개정 전의 것) 있는 경우 제외하고 법령명만 추출
# 법령명 추출용 정규식 (탐욕적 매칭 방지 버전)
_RE_LAW_NAME = re.compile(
    r'(?:구\s+)?'                      # 선택: "구 " 접두사
    r'('
    r'[가-힣]+(?:\s+[가-힣]+){0,5}\s+에\s+관한\s+[가-힣]*(?:법|법률)' # 1. ~에 관한 법률 형태 (최대 6~7단어)
    r'|'
    r'[가-힣]{2,10}\s+[가-힣]{2,10}(?:법|법률|규정|규칙|령|조례)'      # 2. 2단어 형태 (예: 개인정보 보호법)
    r'|'
    r'[가-힣]{2,12}(?:법|법률|규정|규칙|령|조례)'                      # 3. 1단어 형태 (예: 도로교통법)
    r')'
    r'(?:\s*[\(（][^\)）]*[\)）])?'     # 선택: 괄호 부분
    r'(?=\s*제\s*\d+\s*조|\s|$)',       # 뒤에 조항이나 공백/끝
    re.MULTILINE,
)

# 청크 단위 데이터클래스
@dataclass
class Chunk:
    chunk_index: int # 문서 내 순서로.
    text: str
    article: str = ""  # 조   예) "제3조", "제3조의2"
    law_name: str = ""  # 텍스트에서 추출한 법령명  예) "개인정보 보호법"
    char_count: int = field(init=False) #청킹 잘되는지 확인용으로 필요해서 넣었어요.

# __post_init__이 필요한 이유가 text를 기반으로 자동 계산해서 세팅을 진행해야하기때문.
# 들어오는 text에 맞춰서 계산해서 그에따라 세팅하는작업.
    def __post_init__(self):
        self.char_count = len(self.text)
    
    def preview(self, n: int=60) ->str:
        # self.text > n보다 크면 self.text[:n]출력
        # 작으면 self.text 반환
        return self.text[:n]  if len(self.text) > n else self.text

# 청킹 설정
@dataclass
class ChunkConfig:
    max_chars: int = 1000    # BGE-M3 최대 8192토큰, 한국어 기준 1000자 권장
    overlap_chars: int = 100  # 문맥 유지를 위한 청크 간 겹치는 글자 수
    min_chars: int = 10       # 이보다 적으면 청크를 버림
    use_law_structure: bool = True  # 법률 구조 기반 1차 분리 사용 여부 법령명때문에

# 조 정보 추출 헬퍼
# 🎯 [핵심 수정] 뒤에 무엇이 오든 무조건 '제N조' 형태로만 반환
def _extract_article(text: str) -> str:
    '''
    텍스트 앞부분에서 조 번호를 추출하여 '제N조' 형태로만 반환합니다.
    뒤에 붙는 '의 2', '의 규정' 등은 모두 무시하고 깔끔하게 잘라냅니다.
    '''
    m = _RE_ARTICLE_NUM.search(text[:50])
    if not m:
        return ""
    return f"제{m.group(1)}조"

# 법령명 추출 헬퍼 (후처리 및 예외 필터링 추가)
def extract_law_name(text: str) -> str:
    '''
    텍스트 전체에서 법령명을 추출한다.
    글자 수 제한 및 조사 제거 후처리, 예외 단어 필터링이 적용됨.
    '''
    m = _RE_LAW_NAME.search(text)
    if not m:
        return ""
    
    law_name = m.group(1).strip()
    
    # 💡 [방어 코드] "농도를 기초로 도로교통법" 처럼 앞에 조사나 동사가 붙어 나온 경우 잘라내기
    words = law_name.split()
    if len(words) > 1 and "에 관한" not in law_name:
        # 첫 단어가 조사나 어미(~로, ~고, ~에, ~를, ~을, ~의, ~은, ~는)로 끝나면 첫 단어 버림
        if words[0].endswith(('로', '고', '에', '를', '을', '의', '과', '와', '은', '는')):
            law_name = " ".join(words[1:])
            
    # 💡 [예외 필터링] 법으로 끝나지만 법령명이 아닌 것 제외
    EXCLUDE_LIST = ["한글맞춤법", "표준어규정", "외래어표기법"]
    EXCLUDE_SUFFIXES = ('방법', '수법', '기법', '불법', '합법', '위법', '적법', '준법', '탈법', '사법', '입법')

    if law_name in EXCLUDE_LIST or law_name.endswith(EXCLUDE_SUFFIXES):
        return ""
        
    return law_name

# 법률 구조 기반 1차 분리
def _split_by_law_structure(text:str) ->list[str]:
    '''
    조(제N조) 경계로 텍스트를 1차로 분리.
    법률 구조가 없는 일반 문서는 분리없이 전체를 반환
    '''
    sections= _RE_ARTICLE.split(text)

    if len(sections) <=1:
        return [text.strip()] if text.strip() else []
    
    return [s.strip() for s in sections if s.strip()]



# 메인 청킹
class Chunker:
    # chunkconfig에 기본값이 들어갈지 커스텀 할지를 구분하기위해서 None을 넣었습니다.
    # None이면 기본설정값이 수정해야한다면 커스텀해서 넣으면됩니다.
    def __init__(self, config: ChunkConfig | None =None):
        self.config = config or ChunkConfig()
    # 텍스트 정제 
    # unicode 및 한문데이터 변환
    # unicode 제어문자 제거, 특수 공백 → 일반 공백 치환,법률 문서 특수기호 제거
    def _clean(self, text:str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
        text = re.sub(r"[\xa0\u3000\t]", " ", text)
        text = re.sub(r"[\u2022\u25a0\u25cb\u2013\u2014]", "", text)
        text = _hanja2hangul(text)
        return text
    
    # 섹션 하나를 문장 단위로 분리 후 청크 조립
    def _assemble_chunks(
        self,
        sentences: list[str],
        chunk_idx_start: int,
        article:str,
        law_name: str ="",
    )-> list[Chunk]:
        '''
        문장 리스트 → max_chars 기준으로 묶어서 Chunk 리스트 반환.
        overlap도 적용합니다.
        '''
        cfg = self.config
        chunks :list[Chunk] = []
        buffer = ""
        chunk_idx = chunk_idx_start

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            if buffer and len(buffer) + len(sent) + 1 >cfg.max_chars:
                if len(buffer) >=cfg.min_chars:
                    chunks.append(Chunk(
                        chunk_index=chunk_idx,
                        text=buffer.strip(),
                        article=article,
                        law_name=law_name,
                    ))
                    chunk_idx +=1
                
                overlap_text = buffer[-cfg.overlap_chars:] if cfg.overlap_chars else ""
                buffer = overlap_text + " " + sent if overlap_text else sent
            else:
                buffer = buffer + " " + sent if buffer else sent
        
        if buffer.strip() and len(buffer.strip()) >= cfg.min_chars:
            chunks.append(Chunk(
                chunk_index=chunk_idx,
                text=buffer.strip(),
                article=article,
                law_name=law_name,
            ))
        
        return chunks


# 메인청킹부분
    def chunk(self, text:str,already_cleaned: bool = False,on_progress: Callable[[int, int], None] | None = None,  # 콜백 추가
) -> list[Chunk]:
        '''
        텍스트 -> Chunk 리스트 반환
        처리순서
            1. 텍스트 정제 (already_cleaned=True면 스킵)
            2. 법률 구조(조) 기반 1차 분리
            3. 각 섹션별 KSS 문장 분리 (2차)
            4. max_chars 기준 청크 조립 + overlap 적용
            5. 조 메타데이터 각 청크에 부착
        '''
        cfg = self.config
        
        #1. already_cleaned=True면 정제 스킵
        if not already_cleaned:
            text = self._clean(text)
        
        #2. 법률 구조 기반 1차 분리
        if cfg.use_law_structure:
            sections =_split_by_law_structure(text)
        else:
            sections = [text]
        
        all_chunks: list[Chunk] = []
        chunk_idx = 0
        # 나눠진 섹션 갯수 확인.
        total_sections = len(sections)

        # 전체 텍스트에서 법령명 1회 추출(섹션별로 반복 추출x)
        doc_law_name = extract_law_name(text)

        for sec_i, section in enumerate(sections):
            if not section.strip():
                continue
            
            #조 정보 추출
            article = _extract_article(section)

            # 3. KSS 문장 분리 (2차)
            # 섹션이 max_chars 이하인경우 KSS 없이 섹션 그대로 사용한다.
            if len(section) <= cfg.max_chars:
                sentences = [section]
            else:
                # 문장기호에 따라 문단의 나누는 방식 punct
                sentences = _split_sentences(section, backend="punct") 
                if not sentences:
                    sentences = [section]
            
            # 4.청크조립
            new_chunks = self._assemble_chunks(
                sentences= sentences,
                chunk_idx_start = chunk_idx,
                article= article,
                law_name = doc_law_name,
            )
            
            # extend list =[a,a] 가있을시 list.extend([1,2,3]) => [a,a,1,2,3]이런식으로 들어감.
            all_chunks.extend(new_chunks)
            chunk_idx += len(new_chunks)

            #진행률 콜백
            if on_progress and sec_i % 10 ==0:
                on_progress(sec_i, total_sections)
        
        if on_progress:
            on_progress(total_sections, total_sections)
        
        return all_chunks
    
    #청크 확인용.
    def chunk_with_stats(self, text: str) -> dict:
        '''청크 결과 + 통계 변환 (디버깅/검증용)'''
        chunks = self.chunk(text)
        return {
            "total_chunks": len(chunks),
            "total_chars": sum(c.char_count for c in chunks),
            "avg_chars":    sum(c.char_count for c in chunks) / len(chunks) if chunks else 0,
            "min_chars": min(c.char_count for c in chunks) if chunks else 0,
            "max_chars": max(c.char_count for c in chunks) if chunks else 0,
            "chunks": chunks,
        }


