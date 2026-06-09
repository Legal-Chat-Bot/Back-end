#청킹 파이프라인
# 문서 -> 문장 분리(KSS) -> 청크 조립 -> 청크리스트 반환
from dataclasses import dataclass, field
from typing import Callable
import re
import unicodedata
from kss import Kss



# KSS 모듈 초기화(한번만)
_split_sentences = Kss("split_sentences")
# 유저가 올린 문서에 한문데이터 한글로변환
_hanja2hangul = Kss("hanja2hangul")

# 청크 단위 데이터클래스
@dataclass
class Chunk:
    chunk_index: int # 문서 내 순서로.
    text: str
    char_count: int = field(init=False)

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
    max_chars: int = 500 #테스트용 청크글자수
    overlap_chars: int = 50 # 청크 간 겹치는 글자 수(문맥 유지)
    min_chars: int = 10 # 이보다 적으면 청크를 버림.


# 메인 청킹
class Chunker:
    # chunkconfig에 기본값이 들어갈지 커스텀 할지를 구분하기위해서 None을 넣었습니다.
    # None이면 기본설정값이 수정해야한다면 커스텀해서 넣으면됩니다.
    def __init__(self, config: ChunkConfig | None =None):
        self.config = config or ChunkConfig()

    def chunk(self, text:str,already_cleaned: bool = False,    on_progress: Callable[[int, int], None] | None = None,  # 콜백 추가
) -> list[Chunk]:
        '''
        텍스트 -> Chunk 리스트 반환
        1. 한자 -> 한글 변환
        2. KSS 문장 분리
        3. max_chars 기준으로 문장 묶어 청크조립
        4. overlap 적용.
        '''
        cfg = self.config

        #1. already_cleaned=True면 정제 스킵
        if not already_cleaned:
            # unicode 및 한문데이터 변환
            # unicode 제어문자 제거, 특수 공백 → 일반 공백 치환,법률 문서 특수기호 제거
            text = unicodedata.normalize("NFKC", text)
            text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
            text = re.sub(r"[\xa0\u3000\t]", " ", text)
            text = re.sub(r"[\u2022\u25a0\u25cb\u2013\u2014]", "", text)
            text = _hanja2hangul(text)

        # 2. KSS 문장 분리
        sentences = _split_sentences(text, backend="punct")  # punct: 빠른 모드
        if not sentences:
            return []
        
        #3.문장 묶어서 청크조립
        chunks: list[Chunk] = []
        buffer = ""
        chunk_idx =0
        total = len(sentences)




        for i, sent in enumerate(sentences):
            sent = sent.strip()
            if not sent:
                continue
            
            #버퍼 + 문장이 max_chars 초과 시 -> 현재 버퍼를 청크로 확정
            if buffer and len(buffer) + len(sent) + 1 > cfg.max_chars:
                if len(buffer) >= cfg.min_chars:
                    chunks.append(Chunk(chunk_index=chunk_idx, text=buffer.strip()))
                    chunk_idx +=1
                
                # overlap: 이전 청크 끝부분 가져와서 다음 청크 시작에붙임.
                overlap_text = buffer[-cfg.overlap_chars:] if cfg.overlap_chars else ""
                buffer = overlap_text + " " + sent if overlap_text else sent
            else:
                buffer = buffer + " " + sent if buffer else sent
            
            # 콜백 있으면 진행률 전달
            if on_progress and i % 10 == 0:
                on_progress(i, total)
        
        #마지막 남은 버퍼 처리
        if buffer.strip() and len(buffer.strip()) >= cfg.min_chars:
            chunks.append(Chunk(chunk_index=chunk_idx, text=buffer.strip()))
        
        if on_progress:
            on_progress(total, total)  # 완료

        return chunks
    
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


