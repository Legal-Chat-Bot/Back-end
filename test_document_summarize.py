# ============================================================
# test_run.py (문서 추출 및 요약 통합 엔드투엔드 테스트)
#
# 위치: 프로젝트 루트 폴더 (app 폴더와 같은 위치)
# 실행: python test_run.py
# ============================================================

import sys
import os
import types

# 1. 파이썬 패키지 경로에 프로젝트 루트 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 2. [의존성 우회] 실제 app/db/vector/read_ocr.py 나 chunker.py 가 없다면 
#    임포트 에러가 발생하므로, 파일이 없는 경우를 대비해 최소한의 가짜 스텁을 주입합니다.
try:
    from app.db.vector import chunker
except ModuleNotFoundError:
    import re
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
    fake_chunker.Chunker = StubChunker; fake_chunker.ChunkConfig = lambda *a, **k: None; fake_chunker.Chunk = StubChunk
    sys.modules["app.db.vector.chunker"] = fake_chunker

try:
    from app.db.vector import read_ocr
except ModuleNotFoundError:
    fake_ocr = types.ModuleType("app.db.vector.read_ocr")
    fake_ocr.process_pdf = lambda file_bytes, filetype: "PDF/Office 파일에서 가짜로 추출된 텍스트 내용입니다."
    sys.modules["app.db.vector.read_ocr"] = fake_ocr


# 3. 실제 파이프라인 및 요약 모듈 임포트
try:
    from app.db.vector import document_pipeline as dp
    from app.db.vector import document_summarize as ds
    print("✅ 패키지 모듈 로드 성공! (pipeline & summarize)")
except ModuleNotFoundError as e:
    print(f"❌ 임포트 실패: {e}")
    sys.exit(1)


if __name__ == "__main__":
    # --------------------------------------------------------
    # 🔍 [테스트 설정] 여기에 테스트하고 싶은 실제 파일명을 적어주세요!
    # 예: "sample.pdf", "test.docx", "계약서.hwp", "보고서.txt" 등 모든 포맷 지원
    # --------------------------------------------------------
    TARGET_FILE = "sample.pdf" 
    
    print("\n" + "="*50)
    print(f"📂 대상 파일: {TARGET_FILE}")
    print(f"🤖 Ollama 모델: {ds.settings.SUMMARIZE_MODEL}")
    print("="*50)

    # 1. 파일 존재 여부 확인
    if not os.path.exists(TARGET_FILE):
        print(f"❌ 에러: '{TARGET_FILE}' 파일을 찾을 수 없습니다.")
        print(f"프로젝트 루트 폴더({current_dir})에 파일을 올려두었는지 확인하세요.")
        sys.exit(1)

    # 2. 파이프라인 방식: 파일을 바이너리(bytes)로 읽기
    try:
        with open(TARGET_FILE, "rb") as f:
            file_bytes = f.read()
        print(f"💾 파일을 바이너리로 읽어왔습니다. (크기: {len(file_bytes)} bytes)")
    except Exception as e:
        print(f"❌ 파일을 읽는 중 오류 발생: {e}")
        sys.exit(1)

    # 3. 분기 1: document_pipeline을 통해 텍스트 자동 추출
    print("\n[단계 1] document_pipeline 텍스트 추출 시작...")
    print("-" * 40)
    try:
        # 파일 확장자에 맞는 인코딩/디코딩/OCR 분기를 알아서 처리합니다.
        extracted_text = dp.extract_text_from_file(file_bytes, TARGET_FILE)
        print(f"✅ 텍스트 추출 성공! (추출된 총 글자 수: {len(extracted_text)}자)")
        print(f"--- 추출된 텍스트 앞부분 샘플 ---\n{extracted_text[:150]}\n---------------------------------")
    except Exception as e:
        print(f"❌ 텍스트 추출 단계 실패: {e}")
        sys.exit(1)

    # 4. 분기 2: 추출된 텍스트를 요약 모듈로 전달
    print("\n[단계 2] 로컬 Ollama 연동 요약 시작...")
    print("-" * 40)
    try:
        result = ds.summarize_document(extracted_text)
        print(f"📌 최종 카테고리 : {result.category}")
        print(f"📌 정규식 개정 날짜: {result.law_date}")
        print(f"📌 정규식 법령 이름: {result.law_name}")
        print(f"📌 생성된 청크 개수: {result.chunk_count}")
        print(f"\n📝 LLM 요약 내용:\n{result.summary}")
        
        if result.error:
            print(f"\n⚠️ 프로세스 내부 에러 알림: {result.error}")
    except Exception as e:
        print(f"❌ 요약 단계 실행 중 예외 발생: {e}")

    print("\n" + "="*50)
    print("🏁 전체 통합 테스트 프로세스 종료")
    print("="*50)