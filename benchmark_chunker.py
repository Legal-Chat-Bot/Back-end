# ============================================================
# benchmark_chunker.py — KSS 커스텀 청커 vs LangChain 비교
#
# 위치: 프로젝트 루트 폴더
# 실행: python benchmark_chunker.py
# ============================================================

import sys
import os
import time
import types
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# ── 의존성 우회 (test_document_summarize.py 와 동일) ──────────
try:
    from app.db.vector import chunker as _ck
except ModuleNotFoundError:
    from dataclasses import dataclass, field
    @dataclass
    class StubChunk:
        chunk_index: int; text: str; article: str = ""; law_name: str = ""
        def __post_init__(self): self.char_count = len(self.text)
    class StubChunker:
        def chunk(self, text, **kwargs):
            m = re.search(r'([가-힣]{2,10}법)', text)
            return [StubChunk(chunk_index=0, text=text, law_name=m.group(1) if m else "")]
    fake_chunker = types.ModuleType("app.db.vector.chunker")
    fake_chunker.Chunker = StubChunker
    fake_chunker.ChunkConfig = lambda *a, **k: None
    fake_chunker.Chunk = StubChunk
    sys.modules["app.db.vector.chunker"] = fake_chunker

try:
    from app.db.vector import read_ocr
except ModuleNotFoundError:
    fake_ocr = types.ModuleType("app.db.vector.read_ocr")
    fake_ocr.process_pdf = lambda file_bytes, filetype: ""
    sys.modules["app.db.vector.read_ocr"] = fake_ocr

try:
    from app.db.vector import document_pipeline as dp
    from app.db.vector import document_summarize as ds
    from app.db.vector.chunker import Chunker, ChunkConfig
except ModuleNotFoundError as e:
    print(f"❌ 임포트 실패: {e}")
    sys.exit(1)

# ── LangChain 임포트 ──────────────────────────────────────────
# LangChain 1.x부터 텍스트 스플리터가 별도 패키지로 분리됨.
# 버전에 따라 import 경로가 다르므로 순서대로 시도.
LANGCHAIN_OK = False
RecursiveCharacterTextSplitter = None

_import_attempts = [
    ("langchain_text_splitters", "RecursiveCharacterTextSplitter"),  # 1.x 표준 경로
    ("langchain_classic.text_splitter", "RecursiveCharacterTextSplitter"),  # langchain-classic
    ("langchain.text_splitter", "RecursiveCharacterTextSplitter"),  # 0.x 구버전
]

for module_name, cls_name in _import_attempts:
    try:
        module = __import__(module_name, fromlist=[cls_name])
        RecursiveCharacterTextSplitter = getattr(module, cls_name)
        LANGCHAIN_OK = True
        print(f"✅ LangChain 스플리터 로드 성공: {module_name}.{cls_name}")
        break
    except (ImportError, AttributeError):
        continue

if not LANGCHAIN_OK:
    print("⚠️  RecursiveCharacterTextSplitter를 찾을 수 없습니다.")
    print("   pip install langchain-text-splitters 를 시도해보세요.")
    print("   LangChain 결과는 스킵됩니다.\n")

# ── 설정 ──────────────────────────────────────────────────────
TARGET_FILE = "sample.pdf"   # ← 테스트할 파일명 변경
MAX_CHARS   = 1600           # KSS chunker max_chars 와 동일하게 맞춤
REPEAT      = 3              # 측정 반복 횟수 (평균값 사용)

# ── 헬퍼 ──────────────────────────────────────────────────────
def measure(fn, *args, **kwargs):
    """fn을 REPEAT번 실행해서 평균 elapsed(ms)와 결과 반환."""
    elapsed_list = []
    result = None
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed_list.append((time.perf_counter() - t0) * 1000)
    return result, sum(elapsed_list) / len(elapsed_list)

def count_broken_articles(chunks_text_list: list[str]) -> int:
    """조문 중간 절단 횟수: 청크 중간에 '제N조' 패턴이 등장하면 절단된 것."""
    pattern = re.compile(r'제\s*\d+\s*조')
    broken = 0
    for text in chunks_text_list:
        # 첫 글자 이후에 조문 번호가 나타나면 절단된 것으로 간주
        if pattern.search(text[10:]):
            broken += 1
    return broken

def print_section(title: str):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")

def print_row(label, value, unit=""):
    print(f"  {label:<28} {value} {unit}")

# ── 메인 ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print_section("벤치마크 시작")
    print_row("대상 파일", TARGET_FILE)
    print_row("max_chars 설정", MAX_CHARS, "자")
    print_row("반복 횟수", REPEAT, "회")

    # 1. 파일 로드 & 텍스트 추출
    if not os.path.exists(TARGET_FILE):
        print(f"\n❌ '{TARGET_FILE}' 파일 없음. TARGET_FILE 값을 수정하세요.")
        sys.exit(1)

    with open(TARGET_FILE, "rb") as f:
        file_bytes = f.read()

    print_section("단계 1 — 텍스트 추출")
    extracted_text, extract_ms = measure(dp.extract_text_from_file, file_bytes, TARGET_FILE)
    print_row("추출 소요 시간", f"{extract_ms:.1f}", "ms")
    print_row("추출 글자 수",   f"{len(extracted_text):,}", "자")
    print(f"\n  [텍스트 앞 200자 미리보기]")
    print(f"  {extracted_text[:200]}")

    # 2. KSS 커스텀 청커
    print_section("단계 2 — KSS 커스텀 청커")
    kss_chunker = Chunker(ChunkConfig(max_chars=MAX_CHARS))
    kss_chunks, kss_ms = measure(kss_chunker.chunk, extracted_text, clean_text=True)

    kss_texts      = [c.text for c in kss_chunks]
    kss_chars      = [c.char_count for c in kss_chunks]
    kss_with_meta  = sum(1 for c in kss_chunks if c.article)   # 메타데이터 있는 청크
    kss_broken     = count_broken_articles(kss_texts)

    print_row("처리 시간 (평균)",     f"{kss_ms:.1f}",                         "ms")
    print_row("생성된 청크 수",       f"{len(kss_chunks)}",                    "개")
    print_row("평균 청크 길이",       f"{sum(kss_chars)//max(len(kss_chars),1)}","자")
    print_row("최소 / 최대 청크 길이",f"{min(kss_chars)} / {max(kss_chars)}",  "자")
    print_row("조문 메타데이터 보존", f"{kss_with_meta}/{len(kss_chunks)}",     "개")
    print_row("조문 경계 절단 횟수",  f"{kss_broken}",                         "개")

    law_name = kss_chunks[0].law_name if kss_chunks else "(없음)"
    print_row("추출된 법령명",        law_name)

    print(f"\n  [청크 샘플 — 처음 2개]")
    for c in kss_chunks[:2]:
        print(f"  ┌ 청크[{c.chunk_index}] 조문:{c.article or '없음'} / {c.char_count}자")
        print(f"  │ {c.text[:120].replace(chr(10),' ')} ...")
        print(f"  └")

    # 3. LangChain RecursiveCharacterTextSplitter
    print_section("단계 3 — LangChain RecursiveCharacterTextSplitter")
    if LANGCHAIN_OK:
        lc_splitter = RecursiveCharacterTextSplitter(
            chunk_size=MAX_CHARS,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        lc_chunks, lc_ms = measure(lc_splitter.split_text, extracted_text)

        lc_chars  = [len(t) for t in lc_chunks]
        lc_broken = count_broken_articles(lc_chunks)

        print_row("처리 시간 (평균)",     f"{lc_ms:.1f}",                          "ms")
        print_row("생성된 청크 수",       f"{len(lc_chunks)}",                     "개")
        print_row("평균 청크 길이",       f"{sum(lc_chars)//max(len(lc_chars),1)}", "자")
        print_row("최소 / 최대 청크 길이",f"{min(lc_chars)} / {max(lc_chars)}",    "자")
        print_row("조문 메타데이터 보존", "❌ 지원 안 함")
        print_row("조문 경계 절단 횟수",  f"{lc_broken}",                          "개")

        print(f"\n  [청크 샘플 — 처음 2개]")
        for i, t in enumerate(lc_chunks[:2]):
            print(f"  ┌ 청크[{i}] {len(t)}자")
            print(f"  │ {t[:120].replace(chr(10),' ')} ...")
            print(f"  └")
    else:
        print("  ⚠️  LangChain 미설치로 스킵")
        lc_ms     = None
        lc_chunks = []
        lc_broken = None

    # 4. 최종 비교 요약표
    print_section("최종 비교 요약")
    header = f"  {'항목':<26} {'KSS 커스텀':>12}  {'LangChain':>12}"
    print(header)
    print(f"  {'-'*53}")

    def row(label, kss_val, lc_val):
        print(f"  {label:<26} {str(kss_val):>12}  {str(lc_val):>12}")

    row("처리 시간 (ms)",
        f"{kss_ms:.1f}",
        f"{lc_ms:.1f}" if lc_ms else "N/A")
    row("청크 수",
        len(kss_chunks),
        len(lc_chunks) if lc_chunks else "N/A")
    row("평균 길이 (자)",
        sum(kss_chars)//max(len(kss_chars),1),
        f"{sum(lc_chars)//max(len(lc_chars),1)}" if lc_chunks else "N/A")
    row("조문 메타데이터",
        f"✅ {kss_with_meta}개 보존",
        "❌ 없음")
    row("조문 경계 절단",
        f"{kss_broken}건",
        f"{lc_broken}건" if lc_broken is not None else "N/A")
    row("법령명 추출",
        f"✅ {law_name}",
        "❌ 없음")

    print(f"\n{'='*55}")
    print("🏁 벤치마크 완료")
    print(f"{'='*55}")