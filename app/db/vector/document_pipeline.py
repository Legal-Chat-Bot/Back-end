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
            # ✅ 수정 pipeline에서 bytes를 그대로 넘기기
            return process_pdf(file_bytes, filetype=doc_type.value)
        except Exception as e:
            raise RuntimeError(f"[{doc_type.value.upper()}] OCR 텍스트 추출 오류: {e}")

    # ── [분기 2] HWP / HWPX → rhwp 직접 파싱 ──
    elif doc_type in [DocType.HWP, DocType.HWPX]:
        try:
            import tempfile, os
            suffix = f".{doc_type.value}"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                doc = rhwp.parse(tmp_path)
                text = doc.extract_text()
                if not text or not text.strip():
                    raise RuntimeError(
                        "HWP 3.0 이하 포맷은 지원하지 않습니다. "
                        "HWPX 형식으로 변환 후 다시 업로드해주세요."
                    )
                return text
            finally:
                os.unlink(tmp_path)
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"[{doc_type.value.upper()}] 텍스트 추출 오류: {e}")

    # ── [분기 3] TXT ──
    elif doc_type == DocType.TXT:
        for encoding in ["utf-8", "cp949", "euc-kr"]:
            try:
                return file_bytes.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise RuntimeError("[TXT] 지원하는 인코딩 형식이 아닙니다.")

    raise RuntimeError(f"분기 처리 누락된 문서 타입: {doc_type}")