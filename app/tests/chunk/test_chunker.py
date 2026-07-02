import sys
import os
import pytest

# 🛠️ 파이썬 패키지 이름 충돌(app.py와 app폴더)을 강제로 우회하는 코드
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from app.db.vector.chunker import Chunker, ChunkConfig


def test_ut_pdf_06_004_legal_chunking_success():
    """
    [단위테스트결과서 ID: UT-PDF-06-004]
    CASE: SSE 텍스트 추출 및 청킹 처리 이상 무
    목적: chunk() 메서드가 제N조(제목) 형식의 조 구조를 경계로 분리하여
          각 조마다 독립 청크를 생성하고, 조 메타데이터가 올바르게 부착되는지 검증

    [수정 근거]
    - _RE_LEGAL_SECTIONS는 '제N조(제목)' 패턴만 경계로 인식하므로
      샘플 텍스트도 해당 패턴을 반드시 포함해야 3개 섹션으로 분리됨.
    - law_name이 없는 짧은 텍스트에서 extract_law_name()은 ""을 반환하므로
      chunk.article은 "제1조" 형태(law_name 접두사 없음)로 반환됨.
      기존 테스트가 "제1조"와 비교하던 방식은 맞으나, 법령명 추출 결과가
      포함될 수도 있으므로 endswith()로 비교하도록 강건하게 수정.
    - max_chars=500이고 각 조 텍스트가 충분히 짧으므로 조당 1청크가 보장됨.
    """
    config = ChunkConfig(max_chars=500, overlap_chars=50, min_chars=10)
    chunker = Chunker(config)

    # ✅ '제N조(제목)' 패턴을 포함해야 _RE_LEGAL_SECTIONS가 3개 섹션으로 분리함
    sample_legal_text = (
        "제1조(목적) 이 법은 대량의 법률 문서를 정밀하게 파싱하는 것을 목적으로 한다. "
        "제2조(정의) 이 규정에서 사용하는 용어의 정의는 다음과 같다. "
        "제3조(적용범위) 본 규정은 백엔드 데이터 파이프라인 전체에 적용한다."
    )

    chunks = chunker.chunk(
        text=sample_legal_text,
        already_cleaned=False,
        clean_text=True,
        on_progress=None,
    )

    # ✅ 3개 조 → 3개 청크
    assert len(chunks) == 3, f"예상 청크 수는 3개이나, {len(chunks)}개가 생성되었습니다."

    # ✅ _extract_article은 law_name이 없을 경우 "제1조" 형태를 반환함.
    #    law_name이 붙는 경우("형사소송법 제1조")도 포함하기 위해 endswith()로 검증.
    assert chunks[0].article.endswith("제1조"), (
        f"첫 번째 청크 article이 '제1조'로 끝나야 합니다. 실제값: {chunks[0].article!r}"
    )
    assert chunks[1].article.endswith("제2조"), (
        f"두 번째 청크 article이 '제2조'로 끝나야 합니다. 실제값: {chunks[1].article!r}"
    )
    assert chunks[2].article.endswith("제3조"), (
        f"세 번째 청크 article이 '제3조'로 끝나야 합니다. 실제값: {chunks[2].article!r}"
    )

    # ✅ 각 청크 본문에 해당 조의 핵심 키워드가 포함되어야 함
    assert "목적" in chunks[0].text, f"첫 번째 청크에 '목적'이 없습니다: {chunks[0].text!r}"
    assert "적용범위" in chunks[2].text, f"세 번째 청크에 '적용범위'가 없습니다: {chunks[2].text!r}"


def test_chunker_observability_stats():
    """
    CASE: chunk_with_stats 메서드가 chunk()를 정상 호출하여
          통계 데이터를 생성해 내는지 검증

    [수정 근거]
    - chunk_with_stats(text, clean_text)는 내부에서
      self.chunk(text, clean_text=clean_text)를 호출함.
      already_cleaned 인자를 별도로 받지 않으므로 기본값(False) 적용 → 정제 수행.
    - 단일 조 텍스트이므로 섹션 1개 → 청크 1개가 생성되는 것이 정상.
    - 기존 테스트 로직 자체는 올바르나, 호출 시그니처를 명시적으로 검증하는
      주석을 추가하여 향후 시그니처 변경 시 빠르게 인지할 수 있도록 개선.
    """
    config = ChunkConfig(max_chars=100, overlap_chars=10, min_chars=5)
    chunker = Chunker(config)
    sample_text = "제1조(테스트) 본문 내용입니다. 통계 기능 검증을 위한 샘플 데이터입니다."

    # chunk_with_stats(text, clean_text) → 내부: self.chunk(text, clean_text=clean_text)
    stats = chunker.chunk_with_stats(text=sample_text, clean_text=True)

    assert "total_chunks" in stats
    assert "total_chars" in stats
    assert "avg_chars" in stats

    # ✅ 단일 조 텍스트이므로 청크 1개
    assert stats["total_chunks"] == 1, (
        f"단일 조 텍스트에서 청크 1개가 기대되나 {stats['total_chunks']}개가 생성되었습니다."
    )
    assert stats["total_chars"] > 0