"""
형사법 LLM 데이터셋 파이프라인 (인용구 지칭어 노이즈 완벽 숙청판)
=========================================================
개선사항:
  - 🌟 [가짜 법명 및 지칭어 완전 제거] "전단 제38조", "후단 제37조"처럼 법령명 자리에 '전단', '후단', '따라' 같은 조사/지칭어가 온 경우, 리스트 단계에서 하나씩 대조하여 완전히 청소 및 배제
  - [종속/단독 법령 상속] "시행령", "법률" 단독 등장 시 직전 메인 법령명과 안전하게 자동 결합
  - [판결문 로딩 오류 해결] SOURCE_ID_COL["판결문"]을 실제 데이터 헤더인 "판례일련번호"로 일치화
  - CSV 청크 단위 로딩 → 메모리 부족 방지
  - tqdm 진행바 및 kss 한자변환 내장
  - 조인 인덱스 딕셔너리화 (O(1) 조회)
  - [해석례/판결문/결정례 공통 고도화 규칙 반영] 
    1. 모든 인용구는 '조(Article)' 레벨로 변환하여 항/호/목 자동 탈락 및 축소
    2. 법률명 뒤 제개정일 역사 정보 및 약칭 괄호 내용 기호 완벽 제거
    3. 완전 참조와 불완전 참조('법 제2조', '제2조') 혼재 시 불완전 노이즈 제거
    4. "경영하는 학교법인의 임원 중 이사와 감사에 대해서"와 같은 비정형 서술 노이즈를 1차/2차 전수 제거
    5. 추출된 최종 리스트 대상 순서 보존 글로벌 중복 제거
"""

import json
import re
import unicodedata
import logging
from pathlib import Path
from collections import defaultdict
from kss import Kss


import pandas as pd

# ── tqdm 선택적 임포트 ─────────────────────────────────────
# 얼마나 되었는지 확인하기위해 사용했으나 없어도 작동되게함.
try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False
    def tqdm(iterable=None, **kwargs):
        if iterable is None:
            return lambda x: x
        return iter(iterable)
    print("⚠  tqdm 미설치 — pip install tqdm 권장 (진행바 없이 실행)")

# ── kss 설정. ──────────────────────────────────────
_hanja2hangul = Kss("hanja2hangul")
_KSS_AVAILABLE = True



# ── 경로 설정 ──────────────────────────────────────────────
DATA_ROOT   = Path("./data")
SOURCE_DIR  = DATA_ROOT / "Training" / "01.원천데이터"
LABEL_DIR   = DATA_ROOT / "Training" / "02.라벨링데이터"
OUTPUT_FILE = Path("./legal.jsonl")

# CSV 청크 크기 (행 단위)
CSV_CHUNK_SIZE = 50_000

#폴더이름
DOCU_TYPES = ["법령", "판결문", "결정례", "해석례"]
TASK_TYPES = ["_QA", "_SUM"]

#원천데이터 폴더이름 상세.
SOURCE_FOLDER: dict[str, str] = {
    "법령":   "TS_법령",
    "판결문": "TS_판결문",  
    "결정례": "TS_결정례",
    "해석례": "TS_해석례",
}
#라벨폴더 이름
LABEL_FOLDER: dict[str, str] = {
    "법령":   "TL_법령",
    "판결문": "TL_판결문",
    "결정례": "TL_결정례",
    "해석례": "TL_해석례",
}

# 실제 CSV의 컬럼 헤더명인 "판례일련번호"로 일치
SOURCE_ID_COL: dict[str, str] = {
    "법령":   "법령일련번호",
    "판결문": "판례일련번호",  
    "결정례": "결정례일련번호",
    "해석례": "해석례일련번호",
}

#매칭용 라벨과 원천데이터 컬럼.
INFO_ID_COL: dict[str, str] = {
    "법령":   "lawId",
    "판결문": "precedId",
    "결정례": "determintId",
    "해석례": "interpreId",
}

# ── 텍스트 정제 ────────────────────────────────────────────
_RE_BYEOLTAB   = re.compile(r"\[별표\s*\d*\]")
_RE_BYEOLJI    = re.compile(r"\[별지\s*제?\d*호?\s*서식?\]")
_RE_SIHAENG    = re.compile(r"\[시행\s*[\d\.\s]+\]")
_RE_BEOPRYUL   = re.compile(r"\[법률\s*제\d+호\]")
_RE_TABLE      = re.compile(r"<표\s*\d*>")
_RE_FIGURE     = re.compile(r"<그림\s*\d*>")
_RE_WHITESPACE = re.compile(r"\s+")

_CLEAN_PATTERNS = [
    _RE_BYEOLTAB, _RE_BYEOLJI, _RE_SIHAENG,
    _RE_BEOPRYUL, _RE_TABLE,   _RE_FIGURE,
]

#기본 청킹작업 굳이 작성할 필요는 없으나 chunker.py를 안불러오고 사용하기위함.
def clean_text(text: str) -> str:
    """NFKC 정규화 + 법률 문서 특수 마커 제거 + 공백 정규화"""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    for pat in _CLEAN_PATTERNS:
        text = pat.sub("", text)
    text = _RE_WHITESPACE.sub(" ", text).strip()
    return text

#위 로직과 마찬가지이유임.
def convert_hanja(text: str) -> str:
    """kss 한자→한글 변환"""
    if not (_KSS_AVAILABLE and _hanja2hangul and text):
        return text
    try:
        return _hanja2hangul(text)
    except Exception as e:
        return text

#각 날짜가 다 똑같은 형태가 아니기에 동일한 날짜형식으로 변환함.
def format_date(date_str: str) -> str:
    """20221027 → 2022-10-27 / 이미 포맷된 문자열은 그대로"""
    s = str(date_str).strip()
    if re.match(r"^\d{8}$", s):
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return s


def validate_label(record: dict, idx: int, key: str) -> bool:
    """필수 필드 존재 및 output 최소 길이 확인"""
    label = record.get("label", {})
    output = label.get("output", "")
    if not output or len(output.strip()) < 5:
        return False
    if not record.get("info"):
        return False
    return True


# ── 원천 CSV 로딩 ─────────────────────────────────────────
def load_source_csvs(docu_type: str) -> dict[str, list[tuple[float, str]]]:
    folder = SOURCE_FOLDER[docu_type]
    base   = SOURCE_DIR / folder
    if not base.exists():

        return {}

    csv_paths = []
    for p in base.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".csv":
            csv_paths.append(p)
    csv_paths = sorted(csv_paths)

    if not csv_paths:

        return {}

    id_col = SOURCE_ID_COL.get(docu_type)
    index: dict[str, list[tuple[float, str]]] = defaultdict(list)

    for csv_path in tqdm(csv_paths, desc=f"  CSV 로딩 [{docu_type}]", leave=False):
        try:
            chunk_iter = pd.read_csv(
                csv_path,
                encoding="utf-8-sig",
                dtype=str,
                chunksize=CSV_CHUNK_SIZE,
            )
            for chunk in chunk_iter:
                if not id_col or id_col not in chunk.columns:
                    continue
                if "내용" not in chunk.columns:
                    continue
                has_seq = "문장번호" in chunk.columns
                for _, row in chunk.iterrows():
                    rid = str(row[id_col]).strip().zfill(6)
                    seq = float(row["문장번호"]) if has_seq else 0.0
                    content = str(row["내용"]) if pd.notna(row["내용"]) else ""
                    index[rid].append((seq, content))
        except Exception as e:
            print(f"CSV 읽기 실패 {csv_path}: {e}")

    return dict(index)


# ── 라벨 JSON 로딩 ────────────────────────────────────────
def load_label_jsons(docu_type: str, task_type: str) -> list[dict]:
    folder = LABEL_FOLDER[docu_type]
    base   = LABEL_DIR / f"{folder}{task_type}"
    if not base.exists():
        print(f"라벨 경로 없음: {base}")
        return []

    json_paths = sorted(list(base.rglob("*.json")))
    records = []
#tqdm설치했을시 얼마나 되었는지 확인하기위함.
    for json_path in tqdm(json_paths, desc=f"  JSON 로딩 [{docu_type}{task_type}]", leave=False):
        try:
            with open(json_path, encoding="utf-8-sig") as f:
                data = json.load(f)
            if isinstance(data, list):
                records.extend(data)
            elif isinstance(data, dict):
                records.append(data)
        except Exception as e:
            print(f"JSON 읽기 실패 {json_path}: {e}")

    return records


# ── 원천 텍스트 조인 헬퍼 ─────────────────────────────
def get_source_text(
    info: dict,
    docu_type: str,
    source_index: dict[str, list[tuple[float, str]]],
) -> str:
    if not source_index:
        return ""

    info_id_col = INFO_ID_COL.get(docu_type)
    record_id   = str(info.get(info_id_col, "")).strip().zfill(6)
    if not record_id or record_id == "000000":
        return ""

    rows = source_index.get(record_id)
    if not rows:
        return ""

    rows_sorted = sorted(rows, key=lambda x: x[0])
    text = "\n".join(content for _, content in rows_sorted)
    text = clean_text(text)
    text = convert_hanja(text)
    return text


# ── 🌟 해석례/판결문/결정례 다중 조항 및 상속·노이즈 필터링 마스터 헬퍼 ──────────────────
_RE_ARTICLE_PATTERN = r"제\s*\d+\s*조(?:의\s*\d+)?"

def parse_legal_text_metadata(text: str) -> tuple[str, str]:
    if not text:
        return "", ""
        
    sample = text[:800].strip() 
    art_matches = list(re.finditer(_RE_ARTICLE_PATTERN, sample))
    if not art_matches:
        return "", ""
        
    extracted_pairs = []
    current_law = ""
    
    for i, match in enumerate(art_matches):
        start, end = match.span()
        article_str = match.group(0).strip()
        
        prev_end = art_matches[i-1].end() if i > 0 else 0
        between_text = sample[prev_end:start].strip()
        
        # 1. 괄호 제거 및 문장부호 정제
        cleaned_between = re.sub(r"\([^)]+\)", "", between_text)
        cleaned_between = re.sub(r"['’`“”,;·ᆞ•\s및또는]+$", "", cleaned_between).strip()
        cleaned_between = re.sub(r"^['’`“”,;·ᆞ•\s및또는]+", "", cleaned_between).strip()
        
        # 2. 법령명 추출 (기호 유무 및 복합어 통합 탐색)
        law_match = re.search(r"「([^」]+)」$", cleaned_between)
        if law_match:
            current_law = law_match.group(1).strip()
        else:
            compound_law_match = re.search(r"(([가-힣A-Za-z0-9·\-~]+(:\s+[가-힣A-Za-z0-9·\-~]+)*)?(법|법률|규칙|규정|령|칙|조례|헌법))$", cleaned_between)
            if compound_law_match:
                potential_law = compound_law_match.group(1).strip()
                if potential_law not in ["및", "또는", "과", "와", "제", "각"]:
                    current_law = potential_law
            else:
                word_match = re.search(r"([가-힣A-Za-z0-9·\-~]+)$", cleaned_between)
                if word_match:
                    potential_law = word_match.group(1).strip()
                    if potential_law not in ["및", "또는", "과", "와", "제", "각"]:
                        current_law = potential_law
        
        if "경영하는 학교법인의 임원" in current_law:
            current_law = current_law.replace("경영하는 학교법인의 임원 중 이사와 감사에 대해서", "").strip()
            
        # "따라", "전단", "후단" 등 인라인 노이즈 차단 목록
        invalid_law_keywords = [
            "및", "또는", "과", "와", "제", "각", "동", "상기", 
            "따라", "의해", "의한", "종전", "동법", "해당", "전단", "후단"
        ]
        if re.match(r"^제?\d*(항|호|목)$", current_law) or any(kw in current_law for kw in invalid_law_keywords):
            current_law = ""
            
        normalized_art = re.sub(r"\s+", "", article_str)
        extracted_pairs.append((current_law, normalized_art))
            
    # 3. [상속 고도화] 메인 주체 법률명을 기억하여 하위령 및 단독 '법/법률'에 매핑
    resolved_pairs = []
    last_main_law = "" 
    
    for law, art in extracted_pairs:
        is_inc = not law or law in ["법", "법률", "제", "조"] or law.startswith("제") or re.match(r"^제?\d+(조|항|호|목)", law)
        is_suffix_law = law in ["시행령", "시행규칙", "규칙", "령", "동법시행령", "동법시행규칙"]
        is_standalone_law = law in ["법", "법률", "동법"]
        
        if law and not is_inc:
            if is_suffix_law and last_main_law:
                clean_suffix = law.replace("동법", "")
                law = f"{last_main_law} {clean_suffix}"
            elif is_standalone_law and last_main_law:
                law = last_main_law
            else:
                if "시행령" not in law and "시행규칙" not in law:
                    last_main_law = law
            current_law = law
        elif is_inc:
            current_law = ""
            
        effective_law = law if law else current_law
        resolved_pairs.append((effective_law, art))
        
    # 완전 참조 존재 시 불완전 참조 차단
    has_fully_qualified = any(
        l and l not in ["법", "제", "조"] and not l.startswith("제") and not re.match(r"^제?\d+조", l)
        for l, a in resolved_pairs
    )
    
    filtered_citations = []
    unique_laws = []
    
    for l, a in resolved_pairs:
        is_incomplete = not l or l in ["법", "제", "조"] or l.startswith("제") or re.match(r"^제?\d+조", l)
        if has_fully_qualified and is_incomplete:
            continue  
            
        if is_incomplete:
            citation = a
        else:
            citation = f"{l} {a}"
            if l not in unique_laws:
                unique_laws.append(l)
                
        filtered_citations.append(citation)
        
    seen = set()
    final_citations = []
    for c in filtered_citations:
        if c not in seen:
            seen.add(c)
            final_citations.append(c)
            
    if not final_citations:
        return "", ""
        
    return ", ".join(unique_laws), ", ".join(final_citations)


# ──  글로벌 최종 리스트 확인 및 2차 수렴 보정 매크로 ──────────────────
def post_process_clean_meta(text_str: str) -> str:
    """
    조립 완료된 최종 리스트 문자열을 쉼표 단위로 쪼개어 하나씩 확인하며,
    순수 법 규칙으로 끝나지 않는 노이즈('전단', '후단', '따라' 등)가 섞인 가짜 인용구를 완벽하게 제거합니다.
    """
    if not text_str:
        return ""
        
    # 1. 리스트 원소별 하나씩 변경 및 유효성 전수 검사
    items = [item.strip() for item in text_str.split(",") if item.strip()]
    cleaned_items = []
    
    # 순수 법령 지칭으로 인정받을 수 있는 접미사 규칙 정의
    valid_law_suffix = r"(법|법률|규칙|규정|령|칙|조례|헌법|제\d+조(?:의\d+)?)$"
    
    for item in items:
        # 단어 자체가 '전단', '후단' 등으로 시작하거나 포함하는 유해 요소 필터링
        if any(bad in item for bad in ["전단", "후단", "따라", "의해", "종전"]):
            # 만약 "형법 제37조 전단" 처럼 뒤에 꼬리가 붙은 거라면 앞의 순수 법조항만 살려줌
            fix_match = re.search(r"^([가-힣A-Za-z0-9·\-~\s]+제\d+조(?:의\d+)?)", item)
            if fix_match:
                item = fix_match.group(1).strip()
            else:
                continue # 아예 유추 불가능한 가짜는 리스트에서 Drop 탈락
                
        # 최종 단어 형태가 정상적인 법명 포맷 혹은 단독 제N조 형태인지 2차 마감 검사
        tokens = item.split()
        if tokens:
            law_part = tokens[0]
            # 법령명 접미사 체크 혹은 단독 조항 포맷인지 체크
            if re.search(valid_law_suffix, law_part) or re.match(r"^제\d+조", item):
                cleaned_items.append(item)
                
    # 3. 순서 보존 글로벌 중복 제거 후 최종 쉼표 재조립
    final_seen = set()
    final_list = []
    for ci in cleaned_items:
        if ci not in final_seen:
            final_seen.add(ci)
            final_list.append(ci)
            
    return ", ".join(final_list)


# ── 메인 ─────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("형사법 LLM 데이터 파이프라인 시작 (지칭어 노이즈 완결판)")
    print("=" * 60)

    source_indexes: dict[str, dict] = {}
    for dt in DOCU_TYPES:
        source_indexes[dt] = load_source_csvs(dt)
        print(f"  {dt}: {len(source_indexes[dt]):,}개 문서 인덱싱 완료")

    print("\n[2/3] 라벨링 JSON 처리 및 조인...")
    results: list[dict] = []
    stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "skip": 0, "fallback": 0})

    for docu_type in DOCU_TYPES:
        for task_type in TASK_TYPES:
            if docu_type == "법령" and task_type == "_SUM":
                continue

            records = load_label_jsons(docu_type, task_type)
            key = f"{docu_type}/{task_type.lstrip('_')}"
            print(f"  {key}: {len(records):,}건 로드")

            for idx, record in enumerate(
                tqdm(records, desc=f"  처리 [{key}]", unit="건")
            ):
                stats[key]["total"] += 1

                if not validate_label(record, idx, key):
                    stats[key]["skip"] += 1
                    continue

                info  = record.get("info", {})
                label = record.get("label", {})

                source_text = get_source_text(info, docu_type, source_indexes[docu_type])

                label_clean = {
                    k: clean_text(v) if isinstance(v, str) else v
                    for k, v in label.items()
                }

                if not source_text or not source_text.strip():
                    source_text = label_clean.get("output", "")
                    if source_text:
                        stats[key]["fallback"] += 1

                title = info.get("title", "") if info.get("title") else ""
                sm_class = info.get("smClass", "") if info.get("smClass") else ""
                
                if docu_type in ["해석례", "판결문", "결정례"]:
                    parsed_law, parsed_art = parse_legal_text_metadata(source_text)
                    
                    law_name = parsed_law if parsed_law else title
                    article_clause = parsed_art if parsed_art else sm_class
                    
                    if parsed_law:
                        article = article_clause
                    else:
                        if title and sm_class:
                            article = f"{title} {sm_class}".strip()
                        else:
                            article = (title or sm_class or "").strip()
                else:
                    law_name = title
                    if title and sm_class:
                        article = f"{title} {sm_class}".strip()
                    else:
                        article = (title or sm_class or "").strip()

                # 🌟 [2차 리스트 확인 및 필터 변경 변경 마크 가동]
                law_name = post_process_clean_meta(law_name)
                article = post_process_clean_meta(article)

                for date_field in ("promulgDate", "effectDate", "sentenceDate",
                                   "finalDate", "interpreDate"):
                    if date_field in info:
                        info[date_field] = format_date(str(info[date_field]))

                results.append({
                    "info":        info,
                    "label":       label_clean,
                    "docu_type":   docu_type,
                    "task_type":   task_type.lstrip("_"),
                    "law_name":    law_name,  
                    "article":     article,   
                    "source_text": source_text,
                })

    print(f"\n[3/3] JSONL 저장 → {OUTPUT_FILE}")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for row in tqdm(results, desc="  JSONL 저장", unit="건"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print("처리 완료 요약")
    print("=" * 60)
    total_saved    = len(results)
    total_fallback = sum(v["fallback"] for v in stats.values())
    total_input    = sum(v["total"] for v in stats.values())

    print(f"  총 입력 문서   : {total_input:,}건")
    print(f"  총 대체(정답대입): {total_fallback:,}건 💡")
    print(f"  총 저장 청크   : {total_saved:,}건")
    print(f"  출력 위치      : {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()