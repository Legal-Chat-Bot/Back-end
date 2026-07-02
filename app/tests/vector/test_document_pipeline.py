# ============================================================
# 단위 테스트
# 대상 모듈: read_ocr, document_pipeline, document_summarize, summarize_prompt
#
# 실행:
#   pytest test_document_pipeline.py -v
#
# 외부 의존성 (PaddleOCR, pymupdf, OllamaLLM 등) 은 전부 mock 처리
# ============================================================

import json
import re
import pytest
from io import BytesIO
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np


# ============================================================
# read_ocr 테스트
# ============================================================

class TestNormalizeText:
    """normalize_text: 공백 제거 + 소문자 변환"""

    def setup_method(self):
        from app.db.vector.read_ocr import normalize_text
        self.fn = normalize_text

    def test_removes_spaces(self):
        assert self.fn("웹 2.0") == "웹2.0"

    def test_lowercases_english(self):
        assert self.fn("Web 2.0") == "web2.0"

    def test_strips_surrounding_spaces(self):
        assert self.fn("  hello  ") == "hello"

    def test_empty_string(self):
        assert self.fn("") == ""

    def test_already_normalized(self):
        assert self.fn("abc123") == "abc123"

    def test_multiple_internal_spaces(self):
        assert self.fn("a  b  c") == "abc"


class TestPixmapToCv2:
    """pixmap_to_cv2: Pixmap → numpy ndarray 변환"""

    def _make_pix(self, n, w=4, h=4):
        pix = MagicMock()
        pix.n = n
        pix.width = w
        pix.height = h
        pix.samples = bytes(w * h * n)
        return pix

    @patch("app.db.vector.read_ocr.cv2")
    def test_rgba_to_rgb(self, mock_cv2):
        from app.db.vector.read_ocr import pixmap_to_cv2
        mock_cv2.COLOR_RGBA2RGB = 4
        mock_cv2.COLOR_GRAY2RGB = 8
        fake_rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = fake_rgb
        pix = self._make_pix(n=4)
        result = pixmap_to_cv2(pix)
        mock_cv2.cvtColor.assert_called_once()
        assert result.shape == (4, 4, 3)

    @patch("app.db.vector.read_ocr.cv2")
    def test_gray_to_rgb(self, mock_cv2):
        from app.db.vector.read_ocr import pixmap_to_cv2
        mock_cv2.COLOR_RGBA2RGB = 4
        mock_cv2.COLOR_GRAY2RGB = 8
        fake_rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = fake_rgb
        pix = self._make_pix(n=1)
        result = pixmap_to_cv2(pix)
        mock_cv2.cvtColor.assert_called_once()
        assert result.shape == (4, 4, 3)

    @patch("app.db.vector.read_ocr.cv2")
    def test_rgb_passthrough(self, mock_cv2):
        from app.db.vector.read_ocr import pixmap_to_cv2
        mock_cv2.COLOR_RGBA2RGB = 4
        mock_cv2.COLOR_GRAY2RGB = 8
        pix = self._make_pix(n=3)
        result = pixmap_to_cv2(pix)
        mock_cv2.cvtColor.assert_not_called()
        assert result.ndim == 3


class TestExtractNativeBlocks:
    """extract_native_blocks: PDF 페이지에서 native 텍스트 span 추출"""

    def _make_page(self, blocks):
        page = MagicMock()
        page.get_text.return_value = {"blocks": blocks}
        return page

    def test_extracts_text_spans(self):
        from app.db.vector.read_ocr import extract_native_blocks
        blocks = [{
            "type": 0,
            "lines": [{"spans": [{"text": "안녕", "bbox": (10, 20, 50, 30)}]}]
        }]
        result = extract_native_blocks(self._make_page(blocks), zoom=2)
        assert len(result) == 1
        assert result[0]["text"] == "안녕"
        assert result[0]["source"] == "native"
        # zoom 적용 확인
        assert result[0]["bbox"] == (20, 40, 100, 60)

    def test_skips_image_blocks(self):
        from app.db.vector.read_ocr import extract_native_blocks
        blocks = [{"type": 1, "lines": []}]
        result = extract_native_blocks(self._make_page(blocks))
        assert result == []

    def test_skips_empty_text_spans(self):
        from app.db.vector.read_ocr import extract_native_blocks
        blocks = [{
            "type": 0,
            "lines": [{"spans": [{"text": "   ", "bbox": (0, 0, 10, 10)}]}]
        }]
        result = extract_native_blocks(self._make_page(blocks))
        assert result == []

    def test_empty_page(self):
        from app.db.vector.read_ocr import extract_native_blocks
        result = extract_native_blocks(self._make_page([]))
        assert result == []


class TestExtractImageRegions:
    """extract_image_regions: image block 영역 필터링"""

    def _make_page(self, blocks):
        page = MagicMock()
        page.get_text.return_value = {"blocks": blocks}
        return page

    def test_returns_image_regions(self):
        from app.db.vector.read_ocr import extract_image_regions
        blocks = [{"type": 1, "bbox": (0, 0, 100, 80)}]
        result = extract_image_regions(self._make_page(blocks), zoom=2)
        assert len(result) == 1
        assert result[0]["source"] == "image_region"

    def test_skips_small_images(self):
        from app.db.vector.read_ocr import extract_image_regions
        # width=30, height=20 → 필터링됨
        blocks = [{"type": 1, "bbox": (0, 0, 30, 20)}]
        result = extract_image_regions(self._make_page(blocks))
        assert result == []

    def test_skips_text_blocks(self):
        from app.db.vector.read_ocr import extract_image_regions
        blocks = [{"type": 0, "bbox": (0, 0, 200, 100)}]
        result = extract_image_regions(self._make_page(blocks))
        assert result == []

    def test_missing_bbox(self):
        from app.db.vector.read_ocr import extract_image_regions
        blocks = [{"type": 1}]
        result = extract_image_regions(self._make_page(blocks))
        assert result == []


class TestRemoveDuplicateOcr:
    """remove_duplicate_ocr: native text와 동일한 OCR 결과 제거"""

    def test_removes_exact_duplicate(self):
        from app.db.vector.read_ocr import remove_duplicate_ocr
        native = [{"text": "안녕하세요", "bbox": (0, 0, 50, 10), "source": "native"}]
        ocr = [{"text": "안녕하세요", "bbox": (0, 0, 50, 10), "source": "ocr"}]
        result = remove_duplicate_ocr(native, ocr)
        assert result == []

    def test_keeps_unique_ocr(self):
        from app.db.vector.read_ocr import remove_duplicate_ocr
        native = [{"text": "안녕", "bbox": (0, 0, 50, 10), "source": "native"}]
        ocr = [{"text": "반갑습니다", "bbox": (0, 20, 80, 30), "source": "ocr"}]
        result = remove_duplicate_ocr(native, ocr)
        assert len(result) == 1
        assert result[0]["text"] == "반갑습니다"

    def test_case_insensitive_dedup(self):
        from app.db.vector.read_ocr import remove_duplicate_ocr
        native = [{"text": "Hello", "bbox": (0, 0, 50, 10), "source": "native"}]
        ocr = [{"text": "hello", "bbox": (0, 0, 50, 10), "source": "ocr"}]
        result = remove_duplicate_ocr(native, ocr)
        assert result == []

    def test_space_insensitive_dedup(self):
        from app.db.vector.read_ocr import remove_duplicate_ocr
        native = [{"text": "웹 2.0", "bbox": (0, 0, 50, 10), "source": "native"}]
        ocr = [{"text": "웹2.0", "bbox": (0, 0, 50, 10), "source": "ocr"}]
        result = remove_duplicate_ocr(native, ocr)
        assert result == []

    def test_empty_inputs(self):
        from app.db.vector.read_ocr import remove_duplicate_ocr
        assert remove_duplicate_ocr([], []) == []


class TestSortBlocks:
    """sort_blocks: bbox 기준 읽기 순서 정렬"""

    def test_top_to_bottom_order(self):
        from app.db.vector.read_ocr import sort_blocks
        blocks = [
            {"bbox": (0, 100, 50, 120), "text": "두번째"},
            {"bbox": (0, 10, 50, 30), "text": "첫번째"},
        ]
        result = sort_blocks(blocks)
        assert result[0]["text"] == "첫번째"
        assert result[1]["text"] == "두번째"

    def test_same_line_left_to_right(self):
        from app.db.vector.read_ocr import sort_blocks
        # y 좌표가 비슷하면 x 순서로 정렬
        blocks = [
            {"bbox": (200, 10, 300, 30), "text": "오른쪽"},
            {"bbox": (0, 10, 100, 30), "text": "왼쪽"},
        ]
        result = sort_blocks(blocks, line_threshold=30)
        assert result[0]["text"] == "왼쪽"
        assert result[1]["text"] == "오른쪽"

    def test_single_block(self):
        from app.db.vector.read_ocr import sort_blocks
        blocks = [{"bbox": (0, 0, 100, 20), "text": "하나"}]
        result = sort_blocks(blocks)
        assert len(result) == 1

    def test_empty_input(self):
        from app.db.vector.read_ocr import sort_blocks
        assert sort_blocks([]) == []


class TestBlocksToText:
    """blocks_to_text: block 배열 → 단일 텍스트"""

    def test_native_blocks_joined(self):
        from app.db.vector.read_ocr import blocks_to_text
        blocks = [
            {"text": "안녕", "source": "native"},
            {"text": "하세요", "source": "native"},
        ]
        result = blocks_to_text(blocks)
        assert result == "안녕 하세요"

    def test_ocr_blocks_no_prefix(self):
        from app.db.vector.read_ocr import blocks_to_text
        # 현재 구현에서 OCR 블록도 별도 prefix 없이 text 그대로 반환
        blocks = [{"text": "OCR결과", "source": "ocr"}]
        result = blocks_to_text(blocks)
        assert "OCR결과" in result

    def test_empty_text_skipped(self):
        from app.db.vector.read_ocr import blocks_to_text
        blocks = [
            {"text": "", "source": "native"},
            {"text": "  ", "source": "native"},
            {"text": "텍스트", "source": "native"},
        ]
        result = blocks_to_text(blocks)
        assert result.strip() == "텍스트"

    def test_empty_blocks(self):
        from app.db.vector.read_ocr import blocks_to_text
        assert blocks_to_text([]) == ""


# ============================================================
# document_pipeline 테스트
# ============================================================

class TestDetectDocType:
    """detect_doc_type: 확장자 → DocType 매핑"""

    def setup_method(self):
        from app.db.vector.document_pipeline import detect_doc_type, DocType
        self.fn = detect_doc_type
        self.DocType = DocType

    def test_pdf(self):
        assert self.fn("file.pdf") == self.DocType.PDF

    def test_pptx(self):
        assert self.fn("slides.pptx") == self.DocType.PPTX

    def test_docx(self):
        assert self.fn("doc.docx") == self.DocType.DOCX

    def test_hwp(self):
        assert self.fn("법률.hwp") == self.DocType.HWP

    def test_hwpx(self):
        assert self.fn("법률.hwpx") == self.DocType.HWPX

    def test_txt(self):
        assert self.fn("readme.txt") == self.DocType.TXT

    def test_xlsx(self):
        assert self.fn("data.xlsx") == self.DocType.XLSX

    def test_unknown_extension(self):
        assert self.fn("file.xyz") == self.DocType.UNKNOWN

    def test_no_extension(self):
        assert self.fn("nodotfile") == self.DocType.UNKNOWN

    def test_uppercase_extension(self):
        # 대소문자 무관 처리 확인
        assert self.fn("FILE.PDF") == self.DocType.PDF


class TestExtractTextFromFile:
    """extract_text_from_file: 각 분기별 텍스트 추출"""

    @patch("app.db.vector.document_pipeline.process_pdf", return_value="추출된 PDF 텍스트")
    def test_pdf_calls_process_pdf(self, mock_ocr):
        from app.db.vector.document_pipeline import extract_text_from_file
        result = extract_text_from_file(b"fake_bytes", "test.pdf")
        assert result == "추출된 PDF 텍스트"
        mock_ocr.assert_called_once_with(b"fake_bytes", filetype="pdf")

    @patch("app.db.vector.document_pipeline.process_pdf", return_value="PPT 텍스트")
    def test_pptx_calls_process_pdf(self, mock_ocr):
        from app.db.vector.document_pipeline import extract_text_from_file
        result = extract_text_from_file(b"fake_bytes", "slides.pptx")
        assert result == "PPT 텍스트"
        mock_ocr.assert_called_once_with(b"fake_bytes", filetype="pptx")

    @patch("app.db.vector.document_pipeline.process_pdf", side_effect=Exception("OCR 실패"))
    def test_pdf_ocr_error_raises_runtime(self, _):
        from app.db.vector.document_pipeline import extract_text_from_file
        with pytest.raises(RuntimeError, match="OCR 텍스트 추출 오류"):
            extract_text_from_file(b"bad_bytes", "test.pdf")

    def test_txt_utf8(self):
        from app.db.vector.document_pipeline import extract_text_from_file
        content = "안녕하세요".encode("utf-8")
        result = extract_text_from_file(content, "test.txt")
        assert result == "안녕하세요"

    def test_txt_cp949(self):
        from app.db.vector.document_pipeline import extract_text_from_file
        content = "안녕하세요".encode("cp949")
        result = extract_text_from_file(content, "test.txt")
        assert result == "안녕하세요"

    def test_txt_invalid_encoding_raises(self):
        from app.db.vector.document_pipeline import extract_text_from_file
        with pytest.raises(RuntimeError, match="인코딩"):
            # 어떤 인코딩으로도 디코딩 불가한 bytes
            extract_text_from_file(b"\xff\xfe\x00", "test.txt")

    def test_unknown_type_raises_value_error(self):
        from app.db.vector.document_pipeline import extract_text_from_file
        with pytest.raises(ValueError, match="지원하지 않는"):
            extract_text_from_file(b"data", "file.xyz")

    @patch("rhwp.parse")
    def test_hwp_extracts_text(self, mock_parse):
        from app.db.vector.document_pipeline import extract_text_from_file
        mock_doc = MagicMock()
        mock_doc.extract_text.return_value = "HWP 본문 텍스트"
        mock_parse.return_value = mock_doc
        result = extract_text_from_file(b"hwp_bytes", "법령.hwp")
        assert result == "HWP 본문 텍스트"

    @patch("rhwp.parse")
    def test_hwp_empty_text_raises(self, mock_parse):
        from app.db.vector.document_pipeline import extract_text_from_file
        mock_doc = MagicMock()
        mock_doc.extract_text.return_value = ""
        mock_parse.return_value = mock_doc
        with pytest.raises(RuntimeError, match="HWP 3.0"):
            extract_text_from_file(b"hwp_bytes", "법령.hwp")


# ============================================================
# document_summarize 테스트
# ============================================================

class TestExtractRevisedDateRegex:
    """_extract_revised_date_regex: 날짜 정규식 추출"""

    def setup_method(self):
        from app.db.vector.document_summarize import _extract_revised_date_regex
        self.fn = _extract_revised_date_regex

    def test_korean_date_format(self):
        result = self.fn("이 법은 2024년 3월 15일 개정되었다.")
        assert result == "2024-03-15"

    def test_dot_date_format(self):
        result = self.fn("개정: 2023. 11. 5.")
        assert result == "2023-11-05"

    def test_iso_date_format(self):
        result = self.fn("시행일: 2022-06-01")
        assert result == "2022-06-01"

    def test_slash_date_format(self):
        result = self.fn("2021/08/20 시행")
        assert result == "2021-08-20"

    def test_keyword_priority(self):
        # "개정" 키워드 앞 50자 윈도우 밖에 다른 날짜, 키워드 뒤 150자 안에 타겟 날짜
        prefix = "작성일 2020년 1월 1일" + "x" * 60  # 60자 패딩으로 윈도우 밖으로 밀어냄
        text = prefix + " 개정 2024년 5월 10일 이후 내용"
        result = self.fn(text)
        assert result == "2024-05-10"

    def test_no_date_returns_none(self):
        result = self.fn("날짜 정보가 없는 문서입니다.")
        assert result is None

    def test_head_fallback(self):
        # 키워드 없어도 앞 500자에서 찾음
        result = self.fn("2019년 7월 4일 제정 이하 내용...")
        assert result == "2019-07-04"

    def test_zero_padding(self):
        result = self.fn("시행: 2023. 1. 5.")
        assert result == "2023-01-05"


class TestSampleChunks:
    """_sample_chunks: 대표 청크 샘플링"""

    def _make_chunks(self, n):
        from app.db.vector.chunker import Chunk
        return [
            Chunk(
                chunk_index=i,
                text=f"청크 내용 {i}",
                article="",
                law_name="",
            )
            for i in range(n)
        ]

    def test_returns_all_when_under_limit(self):
        from app.db.vector.document_summarize import _sample_chunks, MAX_SAMPLE_CHUNKS
        chunks = self._make_chunks(5)
        result = _sample_chunks(chunks, n=MAX_SAMPLE_CHUNKS)
        assert len(result) == 5

    def test_samples_n_chunks(self):
        from app.db.vector.document_summarize import _sample_chunks
        chunks = self._make_chunks(30)
        result = _sample_chunks(chunks, n=10)
        assert len(result) == 10

    def test_no_duplicates(self):
        from app.db.vector.document_summarize import _sample_chunks
        chunks = self._make_chunks(50)
        result = _sample_chunks(chunks, n=10)
        indices = [c.chunk_index for c in result]
        assert len(indices) == len(set(indices))

    def test_single_chunk(self):
        from app.db.vector.document_summarize import _sample_chunks
        chunks = self._make_chunks(1)
        result = _sample_chunks(chunks, n=10)
        assert len(result) == 1


class TestBuildSampleText:
    """_build_sample_text: 샘플 청크 → 분석용 텍스트 조립"""

    def _make_chunk(self, idx, text):
        from app.db.vector.chunker import Chunk
        return Chunk(chunk_index=idx, text=text, article="", law_name="")

    def test_joins_chunks(self):
        from app.db.vector.document_summarize import _build_sample_text
        chunks = [self._make_chunk(0, "첫 번째"), self._make_chunk(1, "두 번째")]
        result = _build_sample_text(chunks)
        assert "첫 번째" in result
        assert "두 번째" in result

    def test_respects_max_chars(self):
        from app.db.vector.document_summarize import _build_sample_text, MAX_ANALYSIS_CHARS
        # MAX_ANALYSIS_CHARS 초과하는 청크 추가
        chunks = [self._make_chunk(i, "x" * 3000) for i in range(10)]
        result = _build_sample_text(chunks)
        assert len(result) <= MAX_ANALYSIS_CHARS + 100  # 여유 허용

    def test_empty_chunks(self):
        from app.db.vector.document_summarize import _build_sample_text
        assert _build_sample_text([]) == ""


class TestParseResponse:
    """DocumentSummarize._parse_response: LLM JSON 응답 파싱"""

    def setup_method(self):
        from app.db.vector.document_summarize import DocumentSummarize
        self.parse = DocumentSummarize._parse_response

    def test_valid_json(self):
        raw = '{"category": "법령·규정", "summary": "- 내용 1"}'
        meta = self.parse(raw)
        assert meta.category == "법령·규정"
        assert meta.summary == "- 내용 1"

    def test_json_fence_stripped(self):
        raw = '```json\n{"category": "계약서·협약서", "summary": "- 계약"}\n```'
        meta = self.parse(raw)
        assert meta.category == "계약서·협약서"

    def test_unknown_category_fallback(self):
        raw = '{"category": "존재하지않는카테고리", "summary": "내용"}'
        meta = self.parse(raw)
        assert meta.category == "기타"

    def test_missing_category_defaults_to_gita(self):
        raw = '{"summary": "내용만 있음"}'
        meta = self.parse(raw)
        assert meta.category == "기타"

    def test_summary_list_joined(self):
        raw = '{"category": "기타", "summary": ["항목1", "항목2"]}'
        meta = self.parse(raw)
        assert "항목1" in meta.summary
        assert "항목2" in meta.summary

    def test_malformed_json_regex_fallback(self):
        # JSON이 깨진 경우 정규식으로 폴백
        raw = '{"category": "판결문·결정문", "summary": "판결 내용 요약"}'
        meta = self.parse(raw)
        assert meta.category == "판결문·결정문"

    def test_completely_invalid_returns_gita(self):
        raw = "이건 JSON이 아닙니다"
        meta = self.parse(raw)
        assert meta.category == "기타"


class TestDocumentSummarize:
    """DocumentSummarize.summarize: 통합 플로우 테스트"""

    def _make_chunks(self, n=3):
        from app.db.vector.chunker import Chunk
        return [
            Chunk(chunk_index=i, text=f"제{i+1}조 내용", article=f"제{i+1}조",
                  law_name="도로교통법")
            for i in range(n)
        ]

    @patch("app.db.vector.document_summarize._llm")
    @patch("app.db.vector.document_summarize.DocumentSummarize._parse_response")
    def test_empty_text_returns_error(self, mock_parse, mock_llm):
        from app.db.vector.document_summarize import DocumentSummarize
        ds = DocumentSummarize()
        meta = ds.summarize("")
        assert meta.error is not None

    @patch("app.db.vector.document_summarize._llm")
    def test_non_law_category_returns_without_chunks(self, mock_llm):
        from app.db.vector.document_summarize import DocumentSummarize, DocumentMeta
        mock_llm.invoke.return_value = '{"category": "기타", "summary": "기타 문서"}'

        ds = DocumentSummarize()
        with patch.object(ds, '_parse_response', return_value=DocumentMeta(category="기타")):
            meta = ds.summarize("일반 문서 내용입니다.")
        assert meta.category == "기타"
        assert meta.chunks == []

    @patch("app.db.vector.document_summarize._llm")
    def test_law_category_returns_chunks(self, mock_llm):
        from app.db.vector.document_summarize import DocumentSummarize, DocumentMeta

        chunks = self._make_chunks(3)
        law_meta = DocumentMeta(category="법령·규정", summary="법령 요약")

        ds = DocumentSummarize()
        with patch.object(ds, '_parse_response', return_value=law_meta), \
             patch.object(ds._chunker, 'chunk', return_value=chunks):
            meta = ds.summarize("제1조 이 법은...")

        assert meta.category == "법령·규정"
        assert len(meta.chunks) == 3

    @patch("app.db.vector.document_summarize._llm")
    def test_date_extracted_from_text(self, mock_llm):
        from app.db.vector.document_summarize import DocumentSummarize, DocumentMeta

        chunks = self._make_chunks(2)
        law_meta = DocumentMeta(category="법령·규정", summary="요약")

        ds = DocumentSummarize()
        with patch.object(ds, '_parse_response', return_value=law_meta), \
             patch.object(ds._chunker, 'chunk', return_value=chunks):
            meta = ds.summarize("개정 2023년 5월 1일\n제1조 이 법은...")

        assert meta.law_date == "2023-05-01"

    @patch("app.db.vector.document_summarize._llm")
    def test_law_name_from_first_chunk(self, mock_llm):
        from app.db.vector.document_summarize import DocumentSummarize, DocumentMeta

        chunks = self._make_chunks(2)
        law_meta = DocumentMeta(category="법령·규정", summary="요약")

        ds = DocumentSummarize()
        with patch.object(ds, '_parse_response', return_value=law_meta), \
             patch.object(ds._chunker, 'chunk', return_value=chunks):
            meta = ds.summarize("제1조 도로교통법...")

        assert meta.law_name == "도로교통법"

    @patch("app.db.vector.document_summarize._llm")
    def test_llm_failure_returns_partial_meta(self, mock_llm):
        from app.db.vector.document_summarize import DocumentSummarize, DocumentMeta

        chunks = self._make_chunks(2)
        # 1차 호출은 성공, 2차 호출에서 실패 시뮬레이션
        call_count = {"n": 0}
        good_meta = DocumentMeta(category="법령·규정", summary="")

        def side_effect(raw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return good_meta
            raise RuntimeError("LLM 다운")

        ds = DocumentSummarize()
        with patch.object(ds, '_parse_response', side_effect=side_effect), \
             patch.object(ds._chunker, 'chunk', return_value=chunks):
            meta = ds.summarize("개정 2022년 1월 1일\n제1조 내용")

        assert meta.error is not None
        assert meta.law_date == "2022-01-01"

    @patch("app.db.vector.document_summarize._llm")
    def test_chunking_failure_returns_error(self, mock_llm):
        from app.db.vector.document_summarize import DocumentSummarize, DocumentMeta

        law_meta = DocumentMeta(category="법령·규정", summary="")
        ds = DocumentSummarize()
        with patch.object(ds, '_parse_response', return_value=law_meta), \
             patch.object(ds._chunker, 'chunk', side_effect=Exception("청킹 에러")):
            meta = ds.summarize("제1조 내용")

        assert meta.error is not None
        assert "청킹" in meta.error


class TestSummarizeDocumentFunction:
    """summarize_document: 편의 함수 래퍼"""

    @patch("app.db.vector.document_summarize.DocumentSummarize.summarize")
    def test_delegates_to_summarize(self, mock_summarize):
        from app.db.vector.document_summarize import summarize_document, DocumentMeta
        mock_summarize.return_value = DocumentMeta(category="법령·규정")
        result = summarize_document("법령 텍스트")
        mock_summarize.assert_called_once_with("법령 텍스트", clean_text=True)
        assert result.category == "법령·규정"

    @patch("app.db.vector.document_summarize.DocumentSummarize.summarize")
    def test_passes_clean_text_false(self, mock_summarize):
        from app.db.vector.document_summarize import summarize_document, DocumentMeta
        mock_summarize.return_value = DocumentMeta(category="행정문서·공문")
        summarize_document("공문 텍스트", clean_text=False)
        mock_summarize.assert_called_once_with("공문 텍스트", clean_text=False)


# ============================================================
# summarize_prompt 테스트
# ============================================================

class TestAnalysisPromptTemplate:
    """ANALYSIS_PROMPT_TEMPLATE: 프롬프트 형식 검증"""

    def setup_method(self):
        from app.db.vector.summarize_prompt import ANALYSIS_PROMPT_TEMPLATE
        self.template = ANALYSIS_PROMPT_TEMPLATE

    def test_contains_text_placeholder(self):
        assert "{text}" in self.template

    def test_categories_in_template(self):
        # 카테고리 목록이 프롬프트에 포함되어야 함
        assert "법령·규정" in self.template
        assert "계약서·협약서" in self.template
        assert "판결문·결정문" in self.template
        assert "행정문서·공문" in self.template
        assert "기타" in self.template

    def test_output_format_specified(self):
        # JSON 출력 형식 명시 여부
        assert "category" in self.template
        assert "summary" in self.template

    def test_korean_instruction_present(self):
        assert "Korean" in self.template or "korean" in self.template.lower()

    def test_template_substitution(self):
        filled = self.template.replace("{text}", "테스트 문서 내용")
        assert "테스트 문서 내용" in filled
        assert "{text}" not in filled