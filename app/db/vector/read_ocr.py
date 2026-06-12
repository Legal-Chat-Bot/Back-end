import os
import re
import cv2
import numpy as np
import pymupdf

# ============================================================
# PaddleOCR / PaddlePaddle 실행 옵션
# ============================================================
# 중요:
# 이 환경변수들은 반드시 paddleocr import 전에 선언해야 한다.
#
# 네가 겪은 오류:
# NotImplementedError:
# ConvertPirAttribute2RuntimeAttribute not support ...
#
# 이 오류는 PaddleOCR 내부에서 oneDNN / PIR 관련 문제로 발생할 수 있다.
# 그래서 CPU 환경에서는 아래 옵션을 꺼서 안정성을 높인다.
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"


# ============================================================
# OCR 인스턴스 전역 캐싱
# ============================================================
# PaddleOCR 객체는 생성 비용이 크다.
# 매 페이지마다 새로 만들면 매우 느리다.
#
# 따라서 최초 1회만 생성하고 이후에는 재사용한다.
ocr_instance = None


def get_ocr():
    """
    PaddleOCR 인스턴스를 가져오는 함수.

    - 최초 호출 시 PaddleOCR 객체 생성
    - 이후 호출 시 기존 객체 재사용
    - paddleocr import는 반드시 함수 내부에서 한다.
      이유는 파일 상단에서 환경변수를 먼저 설정해야 하기 때문이다.
    """

    global ocr_instance

    if ocr_instance is None:
        from paddleocr import PaddleOCR

        # PaddleOCR 3.x 계열 기준 옵션
        # 사용 중인 버전에 따라 일부 옵션명이 다를 수 있다.
        ocr_instance = PaddleOCR(
            lang="korean",
            device="cpu",

            # 문서 방향 분류 비활성화
            # PDF 슬라이드 OCR에서는 보통 필요하지 않고,
            # 켜두면 속도가 느려질 수 있다.
            use_doc_orientation_classify=False,

            # 문서 펼침 / 왜곡 보정 비활성화
            # 일반 PDF 이미지 영역에서는 필요 없는 경우가 많다.
            use_doc_unwarping=False,

            # 텍스트 라인 방향 감지 비활성화
            # 네 환경에서 OCR 오류가 발생했기 때문에 안정성 우선으로 끈다.
            use_textline_orientation=False,
        )

    return ocr_instance


def normalize_text(text: str) -> str:
    """
    중복 비교용 텍스트 정규화 함수.

    예:
    '웹 2.0' -> '웹2.0'
    ' Web 2.0 ' -> 'web2.0'

    OCR 결과와 native text가 공백 차이만 있는 경우
    중복으로 판단하기 위해 사용한다.
    """

    text = text.lower()
    text = re.sub(r"\s+", "", text)

    return text


def pixmap_to_cv2(pix):
    """
    PyMuPDF Pixmap 객체를 OpenCV에서 사용할 수 있는 numpy 이미지로 변환한다.

    PyMuPDF:
        page.get_pixmap() -> Pixmap

    OpenCV / PaddleOCR:
        numpy.ndarray 형태 이미지 사용

    반환:
        RGB numpy.ndarray
    """

    # Pixmap의 raw pixel 데이터를 numpy 배열로 변환한다.
    img = np.frombuffer(
        pix.samples,
        dtype=np.uint8
    ).reshape(
        pix.height,
        pix.width,
        pix.n
    )

    # pix.n은 채널 수다.
    # 4면 RGBA, 3이면 RGB, 1이면 Gray라고 보면 된다.
    if pix.n == 4:
        # RGBA -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    elif pix.n == 1:
        # Gray -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # PaddleOCR / OpenCV 처리를 위해 메모리 연속 배열로 변환
    return np.ascontiguousarray(img)


def extract_native_blocks(page, zoom=2):
    '''
    PDF 페이지에서 native text를 span 단위로 추출한다.

    native text란?
    - PDF 내부에 실제 텍스트 객체로 들어있는 글자
    - OCR 없이 바로 추출 가능
    - 속도가 빠르고 정확도가 높음

    page.get_text("dict") 구조:
    {
        "blocks": [
            {
                "type": 0,
                "bbox": (...),
                "lines": [
                    {
                        "spans": [
                            {
                                "text": "...",
                                "bbox": (...)
                            }
                        ]
                    }
                ]
            }
        ]
    }

    block type:
    - 0: text block
    - 1: image block

    zoom을 곱하는 이유:
    - OCR 이미지 좌표는 렌더링된 픽셀 좌표 기준이다.
    - page.get_text()의 bbox는 PDF 좌표 기준이다.
    - page.get_pixmap(matrix=Matrix(zoom, zoom))으로 렌더링하면
      이미지 좌표도 zoom배 커진다.
    - 따라서 native text bbox에도 zoom을 곱해서 좌표계를 맞춘다.
    '''

    blocks = []

    data = page.get_text("dict")

    for block in data.get("blocks", []):
        # type 0만 native text
        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()

                # 빈 텍스트는 제외
                if not text:
                    continue

                x0, y0, x1, y1 = span["bbox"]

                blocks.append({
                    # OCR 좌표와 맞추기 위해 zoom 적용
                    "bbox": (
                        x0 * zoom,
                        y0 * zoom,
                        x1 * zoom,
                        y1 * zoom,
                    ),
                    "text": text,
                    "source": "native",
                })

    return blocks


def extract_image_regions(page, zoom=2):
    """
    PDF 페이지에서 image block 영역만 추출한다.

    이 함수의 목적:
    - 페이지 전체 OCR을 하지 않기 위함
    - PDF 안의 이미지 영역만 잘라서 OCR하기 위함

    장점:
    - 전체 페이지 OCR보다 빠름
    - OCR 노이즈 감소
    - native text와 OCR text 병합이 쉬움

    반환값:
    [
        {
            "bbox_pdf": PDF 좌표계 bbox,
            "bbox": 렌더링 이미지 좌표계 bbox,
            "source": "image_region"
        }
    ]
    """

    regions = []

    data = page.get_text("dict")

    for block in data.get("blocks", []):
        # type 1만 image block
        if block.get("type") != 1:
            continue

        bbox = block.get("bbox")

        if not bbox:
            continue

        x0, y0, x1, y1 = bbox

        width = x1 - x0
        height = y1 - y0

        # 너무 작은 이미지 제외
        # 예: 아이콘, 점, 장식 요소 등
        # 이런 것까지 OCR하면 시간만 늘고 결과 품질이 떨어진다.
        if width < 40 or height < 25:
            continue

        regions.append({
            # PDF 원본 좌표
            # page.get_pixmap(clip=...)에는 PDF 좌표가 필요하다.
            "bbox_pdf": (
                x0,
                y0,
                x1,
                y1,
            ),

            # 렌더링 이미지 좌표
            # OCR 결과와 병합할 때 사용한다.
            "bbox": (
                x0 * zoom,
                y0 * zoom,
                x1 * zoom,
                y1 * zoom,
            ),

            "source": "image_region",
        })

    return regions


def render_clip(page, bbox_pdf, zoom=2):
    """
    PDF 페이지의 특정 영역만 이미지로 렌더링한다.

    전체 페이지 렌더링:
        page.get_pixmap(matrix=...)

    특정 영역 렌더링:
        page.get_pixmap(matrix=..., clip=Rect(...))

    여기서는 이미지 block bbox 영역만 잘라서 OCR하기 위해 clip을 사용한다.

    bbox_pdf:
        PDF 좌표계 기준 bbox
        예: (x0, y0, x1, y1)

    반환:
        OpenCV / PaddleOCR에 넣을 수 있는 numpy 이미지
    """

    rect = pymupdf.Rect(bbox_pdf)
    matrix = pymupdf.Matrix(zoom, zoom)

    pix = page.get_pixmap(
        matrix=matrix,

        # clip 영역만 렌더링
        clip=rect,

        # alpha=False로 배경 투명도 제거
        alpha=False,

        # 주석/annotation은 제외
        annots=False,
    )

    return pixmap_to_cv2(pix)


def extract_ocr_blocks_from_image(img, offset_x=0, offset_y=0):
    """
    이미지 하나에 OCR을 실행하고, OCR 결과를 block 구조로 반환한다.

    중요한 점:
    - 여기서 img는 전체 페이지 이미지가 아니다.
    - image block 영역만 잘라낸 clip 이미지다.

    OCR 결과 좌표:
    - OCR이 반환하는 좌표는 clip 이미지 내부 좌표다.
    - 하지만 native text와 병합하려면 전체 페이지 기준 좌표가 필요하다.
    - 그래서 offset_x, offset_y를 더해서 원래 페이지 좌표로 되돌린다.

    예:
    이미지 영역이 페이지에서 x=200, y=100에 있었고
    OCR 결과가 clip 내부에서 x=10, y=20이면
    실제 페이지 기준 좌표는 x=210, y=120이 된다.
    """

    try:
        # PaddleOCR 3.x 계열 predict 방식
        results = get_ocr().predict(img)

    except Exception as e:
        # OCR이 실패해도 전체 API가 죽으면 안 된다.
        # native text만으로도 요약이 가능해야 한다.
        print(f"[OCR 오류] {e}")
        return []

    blocks = []

    for page_result in results:
        # PaddleOCR 3.x 결과 구조
        rec_texts = page_result.get("rec_texts", [])
        rec_polys = page_result.get("rec_polys", [])

        for text, poly in zip(rec_texts, rec_polys):
            text = text.strip()

            if not text:
                continue

            # poly는 OCR 텍스트 영역의 사각형 또는 다각형 좌표다.
            # bbox 정렬을 위해 감싸는 사각형으로 변환한다.
            poly = np.array(poly)

            # clip 내부 좌표 + 원래 이미지 영역 offset
            x0 = float(np.min(poly[:, 0])) + offset_x
            y0 = float(np.min(poly[:, 1])) + offset_y
            x1 = float(np.max(poly[:, 0])) + offset_x
            y1 = float(np.max(poly[:, 1])) + offset_y

            blocks.append({
                "bbox": (
                    x0,
                    y0,
                    x1,
                    y1,
                ),
                "text": text,
                "source": "ocr",
            })

    return blocks


def remove_duplicate_ocr(native_blocks, ocr_blocks):
    """
    native text와 OCR text가 중복되는 경우 OCR 결과를 제거한다.

    왜 필요한가?
    - PDF 이미지 영역 안에 native text처럼 보이는 요소가 있을 수 있다.
    - 또는 OCR이 native text와 같은 내용을 다시 읽을 수 있다.
    - 중복 텍스트가 요약 모델에 들어가면 요약 품질이 떨어진다.

    단순 비교 방식:
    - 공백 제거
    - 소문자 변환
    - 완전히 같은 텍스트면 중복으로 판단
    """

    native_texts = set()

    for block in native_blocks:
        native_texts.add(
            normalize_text(block["text"])
        )

    filtered = []

    for block in ocr_blocks:
        text = normalize_text(block["text"])

        if text in native_texts:
            continue

        filtered.append(block)

    return filtered


def sort_blocks(blocks, line_threshold=20):
    """
    bbox 위치 기준으로 block을 읽기 순서에 가깝게 정렬한다.

    정렬 방식:
    1. y 좌표 기준으로 위에서 아래 정렬
    2. 비슷한 y 좌표끼리는 같은 줄로 묶기
    3. 같은 줄 안에서는 x 좌표 기준으로 왼쪽에서 오른쪽 정렬

    line_threshold:
    - 같은 줄로 판단할 y 좌표 차이
    - 값이 작으면 줄 분리가 촘촘해지고
    - 값이 크면 서로 다른 줄도 같은 줄로 묶일 수 있다.

    슬라이드 PDF는 글자 크기가 크고 영역이 넓으므로
    15~25 정도부터 테스트하는 것을 추천한다.
    """

    # 먼저 y, x 기준으로 대략 정렬
    blocks = sorted(
        blocks,
        key=lambda b: (
            b["bbox"][1],
            b["bbox"][0],
        )
    )

    lines = []

    for block in blocks:
        y0 = block["bbox"][1]
        y1 = block["bbox"][3]

        # block의 세로 중앙값
        center_y = (y0 + y1) / 2

        placed = False

        for line in lines:
            # 현재 line에 있는 block들의 세로 중앙값 평균
            line_center_y = np.mean([
                (b["bbox"][1] + b["bbox"][3]) / 2
                for b in line
            ])

            # y 중앙값이 비슷하면 같은 줄로 판단
            if abs(center_y - line_center_y) < line_threshold:
                line.append(block)
                placed = True
                break

        # 기존 line에 들어가지 못하면 새 line 생성
        if not placed:
            lines.append([block])

    sorted_blocks = []

    for line in lines:
        # 같은 줄 안에서는 왼쪽에서 오른쪽 순서
        line = sorted(
            line,
            key=lambda b: b["bbox"][0]
        )

        sorted_blocks.extend(line)

    return sorted_blocks


def blocks_to_text(blocks):
    """
    정렬된 block 배열을 하나의 텍스트로 변환한다.

    source가 ocr인 경우 [이미지 OCR] 표시를 붙인다.
    나중에 디버깅할 때 어떤 텍스트가 OCR에서 온 것인지 확인하기 좋다.

    실제 요약 모델에 넣을 때 표시가 거슬리면
    '[이미지 OCR]' 부분은 제거해도 된다.
    """

    texts = []

    for block in blocks:
        text = block.get("text", "").strip()

        if not text:
            continue

        if block.get("source") == "ocr":
            texts.append(f"{text}")
        else:
            texts.append(text)

    return " ".join(texts)


def process_pdf(pdf_bytes: bytes, zoom=2, filetype: str = "pdf"):
    """
    PDF 전체 처리 함수.

    입력:
        pdf_bytes:
            FastAPI에서 await file.read()로 읽은 파일 bytes.
            PDF뿐 아니라 pymupdf가 지원하는 모든 포맷(pptx, docx, hwp 등) 가능.

        zoom:
            PDF 렌더링 확대 비율.
            OCR 품질을 위해 2 정도 추천.

        filetype:
            pymupdf.open()에 전달할 파일 타입 힌트.
            기본값 "pdf". pptx/docx/hwp 등도 직접 지정 가능.

    처리 흐름:
        1. PDF 열기
        2. 페이지 반복
        3. native text 추출
        4. image block 영역 추출
        5. image block만 clip 렌더링
        6. clip 이미지 OCR
        7. native text + OCR text 병합
        8. bbox 기준 정렬
        9. 페이지별 document 생성

    반환:
        [
            {
                "page": 1,
                "items": [...],
                "text": "..."
            },
            ...
        ]
    """

    doc = pymupdf.open(
        stream=pdf_bytes,
        filetype=filetype
    )

    document = []

    for page_idx, page in enumerate(doc):
        page_number = page_idx + 1

        print(f"\n===== PAGE {page_number} =====")

        # =========================
        # 1. native text 추출
        # =========================
        # PDF 내부에 텍스트 객체로 들어있는 글자를 추출한다.
        # OCR보다 빠르고 정확하다.
        native_blocks = extract_native_blocks(
            page,
            zoom=zoom
        )

        print("native:", len(native_blocks))

        # =========================
        # 2. image block 영역 추출
        # =========================
        # PDF 안에 포함된 이미지 영역만 가져온다.
        # 이 영역만 OCR하면 전체 페이지 OCR보다 훨씬 빠르다.
        image_regions = extract_image_regions(
            page,
            zoom=zoom
        )

        print("image regions:", len(image_regions))

        # =========================
        # 3. 이미지 영역별 OCR 실행
        # =========================
        ocr_blocks = []

        for region in image_regions:
            try:
                # PDF 좌표계 기준 image bbox
                bbox_pdf = region["bbox_pdf"]

                # 렌더링 이미지 좌표계 기준 image bbox
                bbox_rendered = region["bbox"]

                # image block 영역만 잘라서 이미지로 렌더링
                clip_img = render_clip(
                    page,
                    bbox_pdf,
                    zoom=zoom
                )

                # clip 이미지가 원래 페이지에서 시작하는 좌표
                # OCR 결과 좌표를 전체 페이지 좌표로 되돌릴 때 사용한다.
                x0, y0, _, _ = bbox_rendered

                region_ocr_blocks = extract_ocr_blocks_from_image(
                    clip_img,
                    offset_x=x0,
                    offset_y=y0,
                )

                ocr_blocks.extend(region_ocr_blocks)

            except Exception as e:
                # 특정 이미지 영역 OCR이 실패해도 전체 처리는 계속한다.
                print(f"[이미지 영역 OCR 실패] page={page_number}, error={e}")

        print("ocr before duplicate remove:", len(ocr_blocks))

        # =========================
        # 4. OCR 중복 제거
        # =========================
        # native text와 동일한 OCR 텍스트는 제거한다.
        ocr_blocks = remove_duplicate_ocr(
            native_blocks,
            ocr_blocks
        )

        print("ocr after duplicate remove:", len(ocr_blocks))

        # =========================
        # 5. native + OCR 병합
        # =========================
        merged_blocks = native_blocks + ocr_blocks

        # =========================
        # 6. 위치 기준 정렬
        # =========================
        merged_blocks = sort_blocks(
            merged_blocks,
            line_threshold=20
        )

        # =========================
        # 7. 페이지 텍스트 생성
        # =========================
        page_text = blocks_to_text(merged_blocks)

        # =========================
        # 8. 페이지 결과 저장
        # =========================
        document.append(page_text)

    # =========================
    # 9. 반환 형식 변환
    # =========================
    # 모든 페이지 처리가 끝난 뒤 한 번만 실행해야 한다.
    # for 루프 안에 있으면 매 페이지마다 doc_text_result를 처음부터
    # 재생성하게 되어 불필요한 연산이 반복된다.
    doc_text_result = ""

    for i, page_text in enumerate(document):
        doc_text_result += "\n--- " + str(i + 1) + " Page ---\n"
        doc_text_result += page_text.replace(',', '')

    doc.close()

    return doc_text_result


# def document_to_text(document):
#     """
#     document 배열을 요약 모델에 넣기 좋은 문자열로 변환한다.

#     document 구조:
#     [
#         {
#             "page": 1,
#             "items": [...],
#             "text": "..."
#         }
#     ]

#     반환:
#     [페이지 1]
#     ...
    
#     [페이지 2]
#     ...
#     """

#     lines = []

#     for page in document:
#         page_number = page.get("page")
#         text = page.get("text", "")

#         lines.append(f"[페이지 {page_number}]")
#         lines.append(text)

#     return "\n\n".join(lines).strip()