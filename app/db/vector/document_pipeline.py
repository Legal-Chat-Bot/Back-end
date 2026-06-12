# ============================================================
# 문서 → 확장자 판별 → 텍스트 추출 (read_ocr 연동) 파이프라인
#
# 지원 포맷: pdf, pptx, docx, hwp, hwpx, txt,xlsx
#
# 추출 전략:
#   PDF / PPTX / DOCX / xlsx → pymupdf.open(stream, filetype) → process_pdf()
#   HWP,HWPX (구형 바이너리)     → 1순위: rhwp를 사용하여 hwp-> 텍스트추출
#   TXT                       → decode()
# ============================================================

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from io import BytesIO
import pymupdf
import rhwp  # rhwp-python

from app.db.vector.read_ocr import process_pdf


class DocType(str, Enum):
    PDF     = "pdf"
    PPT     = "ppt"
    PPTX    = "pptx"
    DOCX    = "docx"
    HWP     = "hwp"
    HWPX    = "hwpx"
    TXT     = "txt"
    XLSX    = "xlsx"
    UNKNOWN = "unknown"


def detect_doc_type(filename: str) -> DocType:
    ext = filename.split(".")[-1].lower() if "." in filename else ""
    mapping = {
        "pdf": DocType.PDF, "pptx": DocType.PPTX, "docx": DocType.DOCX,
        "hwp": DocType.HWP, "hwpx": DocType.HWPX, "ppt": DocType.PPT,
        "txt": DocType.TXT,  "xlsx": DocType.XLSX,
    }
    return mapping.get(ext, DocType.UNKNOWN)


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    doc_type = detect_doc_type(filename)

    if doc_type == DocType.UNKNOWN:
        raise ValueError(f"지원하지 않는 파일 형식입니다. (파일명: {filename})")

    # ── [분기 1] PDF, PPTX, DOCX, XLSX, PPT → PyMuPDF + OCR ──
    if doc_type in [DocType.PDF, DocType.PPTX, DocType.DOCX, DocType.XLSX, DocType.PPT]:
        try:
            fitz_doc = pymupdf.open(stream=BytesIO(file_bytes), filetype=doc_type.value)
            # ✅ 수정 pipeline에서 bytes를 그대로 넘기기
            return process_pdf(file_bytes, filetype=doc_type.value)
        except Exception as e:
            raise RuntimeError(f"[{doc_type.value.upper()}] OCR 텍스트 추출 오류: {e}")

    # ── [분기 2] HWP / HWPX → rhwp 직접 파싱 ──
    elif doc_type in [DocType.HWP, DocType.HWPX]:
        try:
            # rhwp는 파일 경로만 받으므로 임시 파일로 저장 후 파싱
            import tempfile, os
            suffix = f".{doc_type.value}"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                doc = rhwp.parse(tmp_path)
                return doc.extract_text()
            finally:
                os.unlink(tmp_path)
        except Exception as e:
            raise RuntimeError(f"[{doc_type.value.upper()}] rhwp 텍스트 추출 오류: {e}")

    # ── [분기 3] TXT ──
    elif doc_type == DocType.TXT:
        for encoding in ["utf-8", "cp949", "euc-kr"]:
            try:
                return file_bytes.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise RuntimeError("[TXT] 지원하는 인코딩 형식이 아닙니다.")

    raise RuntimeError(f"분기 처리 누락된 문서 타입: {doc_type}")