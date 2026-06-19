'''
=========================================================
형사법 LLM 데이터셋 파이프라인 인용구 지칭어 노이즈 개선판
=========================================================
개선사항:
- [가짜 법명 및 지칭어 완전 제거] "전단 제38조", "후단 제37조"처럼 법령명 자리에 '전단', '후단', '따라' 같은 조사/지칭어가 온 경우, 리스트 단계에서 하나씩 대조하여 완전히 청소 및 배제
- [종속/단독 법령 상속] "시행령", "법률" 단독 등장 시 직전 메인 법령명과 안전하게 자동 결합
- [판결문 로딩 오류 해결] SOURCE_ID_COL["판결문"]을 실제 데이터 헤더인 "판례일련번호"로 일치화.
- 다른 Doc_type도 일치하지않는 법령일련번호 이런부분이 일치하지 않는 부분이 존재해 SOURCE_ID_COL,INFO_ID_COL 컬럼매칭용 만듬.
- CSV 청크 단위 로딩 → 메모리 부족 방지 => OOM으로 인한 실행불가가 발생해서 청크단위로 진행했습니다. 한번에하면 OOM나서.
- tqdm 진행바 및 kss 한자변환 내장 => tqdm은 설치해도 되고 안되는식으로 구현했습니다. 일단 저는 얼마나 진행되었는지 확인을 위해 설치했습니다.
- 조인 인덱스 딕셔너리화 (O(1) 조회) => OOM문제 해결방법 입니다. 원래는 (O(n**2))방식으로 반복문을 돌렸습니다.
- [해석례/판결문/결정례 공통 고도화 규칙 반영]

1. 모든 인용구는 '조(Article)' 레벨로 변환하여 항/호/목 자동 탈락 및 축소
2. 법률명 뒤 제개정일 역사 정보 및 약칭 괄호 내용 기호 완벽 제거
3. 완전 참조와 불완전 참조('법 제2조', '제2조') 혼재 시 불완전 노이즈 제거
4. 추출된 최종 리스트 대상 순서 보존 글로벌 중복 제거
'''
import json
import re
import unicodedata
from pathlib import Path
from collections import defaultdict
from kss import Kss

import pandas as pd

# ── tqdm 선택적 임포트 ─────────────────────────────────────
try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    #해당 함수의 반환 타입을 (x: Unknown) -> Unknown으로 잘못 추론한것 떄문에 나는 에러수정.
    def tqdm(iterable, *args, **kwargs):
        return iterable
    print("⚠  tqdm 미설치 — pip install tqdm 권장 (진행바 없이 실행)")
# ── kss 고정 설정 ──────────────────────────────
_hanja2hangul = Kss("hanja2hangul")

# ── 경로 및 크기 설정 ───────────────────────────────────────
DATA_ROOT   = Path("./data")
SOURCE_DIR  = DATA_ROOT / "Training" / "01.원천데이터"
LABEL_DIR   = DATA_ROOT / "Training" / "02.라벨링데이터"
OUTPUT_FILE = Path("./legal.jsonl")

CSV_CHUNK_SIZE = 50_000

DOCU_TYPES = ["법령", "판결문", "결정례", "해석례"]
TASK_TYPES = ["_QA", "_SUM"]

SOURCE_FOLDER: dict[str, str] = {
    "법령":   "TS_법령",
    "판결문": "TS_판결문",  
    "결정례": "TS_결정례",
    "해석례": "TS_해석례",
}
LABEL_FOLDER: dict[str, str] = {
    "법령":   "TL_법령",
    "판결문": "TL_판결문",
    "결정례": "TL_결정례",
    "해석례": "TL_해석례",
}

SOURCE_ID_COL: dict[str, str] = {
    "법령":   "법령일련번호",
    "판결문": "판례일련번호",  
    "결정례": "결정례일련번호",
    "해석례": "해석례일련번호",
}

INFO_ID_COL: dict[str, str] = {
    "법령":   "lawId",
    "판결문": "precedId",
    "결정례": "determintId",
    "해석례": "interpreId",
}

# ── 글로벌 컴파일 정규식 목록 ──────────────────────────────────────
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

# ── 텍스트 메타 파싱을 위한 컴파일 정규식 ──
_RE_ARTICLE_PATTERN     = re.compile(r"제\s*\d+\s*조(?:의\s*\d+)?")
_RE_PARENTHESES         = re.compile(r"\([^)]+\)")
_RE_CLEAN_PUNCT_SUFFIX  = re.compile(r"['’`“”,;·ᆞ•\s및또는]+$")
_RE_CLEAN_PUNCT_PREFIX  = re.compile(r"^['’`“”,;·ᆞ•\s및또는]+")
_RE_BRACKETED_LAW       = re.compile(r"「([^」]+)」$")
_RE_MULTIPLE_SPACES     = re.compile(r"\s+")

_RE_COMPOUND_LAW = re.compile(
    r"(([가-힣A-Za-z0-9·\-~]+(:\s+[가-힣A-Za-z0-9·\-~]+)*)?(법|법률|규칙|규정|령|칙|조례|헌법))$"
)
_RE_WORD_LAW = re.compile(r"([가-힣A-Za-z0-9·\-~]+)$")

_RE_INVALID_LAW_MATCH = re.compile(r"^제?\d*(항|호|목)$")
_RE_INCOMPLETE_LAW    = re.compile(r"^제?\d+조")

_RE_CLEAN_POST_FIX   = re.compile(r"^([가-힣A-Za-z0-9·\-~\s]+제\d+조(?:의\d+)?)")
_RE_VALID_LAW_SUFFIX = re.compile(r"(법|법률|규칙|규정|령|칙|조례|헌법|제\d+조(?:의\d+)?)$")
_RE_ARTICLE_START    = re.compile(r"^제\d+조")


# ── 세부 단위 변환/검증/정제 함수식 추출 ────────────────

def clean_and_normalize_punc(text: str) -> str:
    """텍스트의 불필요한 괄호 및 기호, 전후 수식 전치사들을 정밀 청소합니다."""
    text = _RE_PARENTHESES.sub("", text)
    text = _RE_CLEAN_PUNCT_SUFFIX.sub("", text).strip()
    text = _RE_CLEAN_PUNCT_PREFIX.sub("", text).strip()
    return text


def extract_potential_law_name(cleaned_text: str) -> str:
    """텍스트로부터 가장 적합한 법령명을 순차적 우선순위 룰셋으로 추출합니다."""
    law_match = _RE_BRACKETED_LAW.search(cleaned_text)
    if law_match:
        return law_match.group(1).strip()
        
    compound_law_match = _RE_COMPOUND_LAW.search(cleaned_text)
    if compound_law_match:
        potential = compound_law_match.group(1).strip()
        if potential not in ["및", "또는", "과", "와", "제", "각"]:
            return potential
            
    word_match = _RE_WORD_LAW.search(cleaned_text)
    if word_match:
        potential = word_match.group(1).strip()
        if potential not in ["및", "또는", "과", "와", "제", "각"]:
            return potential
            
    return ""


def validate_extracted_law(law_name: str) -> bool:
    """추출된 법령 이름에 유해 지칭어 노이즈가 포함되어 있는지 철저히 검증합니다."""
    if not law_name:
        return False
        
    if _RE_INVALID_LAW_MATCH.match(law_name):
        return False
        
    invalid_keywords = [
        "및", "또는", "과", "와", "제", "각", "동", "상기", 
        "따라", "의해", "의한", "종전", "동법", "해당", "전단", "후단"
    ]
    if any(kw in law_name for kw in invalid_keywords):
        return False
        
    return True


def inherit_and_resolve_law_hierarchy(extracted_pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """'시행령', '동법', '법률' 등의 단독 노출 형태에 대해 직전 상위 법령명을 논리적 결함 없이 안전하게 상속합니다."""
    resolved_pairs = []
    last_main_law = ""
    
    for law, art in extracted_pairs:
        # 상태 판별식 명확화
        is_suffix_law = law in ["시행령", "시행규칙", "규칙", "령", "동법시행령", "동법시행규칙"]
        is_standalone_law = law in ["법", "법률", "동법"]
        is_incomplete_ref = not law or law in ["제", "조"] or law.startswith("제") or bool(_RE_INCOMPLETE_LAW.match(law))
        
        effective_law = law
        
        # 상속 및 결합 판단 트리 구조화
        if is_suffix_law and last_main_law:
            clean_suffix = law.replace("동법", "").strip()
            effective_law = f"{last_main_law} {clean_suffix}"
        elif is_standalone_law and last_main_law:
            effective_law = last_main_law
        elif is_incomplete_ref and last_main_law:
            effective_law = last_main_law
        else:
            # 완결된 새로운 메인 법명이 들어온 경우 갱신
            if law and not is_suffix_law and not is_standalone_law and not is_incomplete_ref:
                if "시행령" not in law and "시행규칙" not in law:
                    last_main_law = law
                effective_law = law
                
        resolved_pairs.append((effective_law, art))
        
    return resolved_pairs


def filter_citations_and_deduplicate(resolved_pairs: list[tuple[str, str]]) -> tuple[str, str]:
    """불완전 인용 노이즈를 쳐내고, 순서가 보존되는 완전 고유 인용 리스트를 반환합니다."""
    has_fully_qualified = any(
        l and l not in ["법", "제", "조"] and not l.startswith("제") and not _RE_INCOMPLETE_LAW.match(l)
        for l, a in resolved_pairs
    )
    
    filtered_citations = []
    unique_laws = []
    
    for l, a in resolved_pairs:
        is_incomplete = not l or l in ["법", "제", "조"] or l.startswith("제") or _RE_INCOMPLETE_LAW.match(l)
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


# ── 원본 메타 헬퍼 (단위 함수식 통합) ───────────────────────────────────

def parse_legal_text_metadata(text: str) -> tuple[str, str]:
    """텍스트 선두영역에서 모든 단위 유틸리티 함수들을 조합하여 최종 메타데이터를 파싱합니다."""
    if not text:
        return "", ""
        
    sample = text[:800].strip() 
    art_matches = list(_RE_ARTICLE_PATTERN.finditer(sample))
    if not art_matches:
        return "", ""
        
    extracted_pairs = []
    
    for i, match in enumerate(art_matches):
        start, end = match.span()
        article_str = match.group(0).strip()
        
        prev_end = art_matches[i-1].end() if i > 0 else 0
        between_text = sample[prev_end:start].strip()
        
        cleaned_between = clean_and_normalize_punc(between_text)
        current_law = extract_potential_law_name(cleaned_between)
            
        if not validate_extracted_law(current_law):
            current_law = ""
            
        normalized_art = _RE_MULTIPLE_SPACES.sub("", article_str)
        extracted_pairs.append((current_law, normalized_art))
            
    resolved_pairs = inherit_and_resolve_law_hierarchy(extracted_pairs)
    return filter_citations_and_deduplicate(resolved_pairs)


def post_process_clean_meta(text_str: str) -> str:
    """조립이 끝난 결과에 지칭어 꼬리가 남은 불순 건(예: 전단)을 전수 2차 수렴 필터링합니다."""
    if not text_str:
        return ""
        
    items = [item.strip() for item in text_str.split(",") if item.strip()]
    cleaned_items = []
    
    for item in items:
        if any(bad in item for bad in ["전단", "후단", "따라", "의해", "종전"]):
            fix_match = _RE_CLEAN_POST_FIX.search(item)
            if fix_match:
                item = fix_match.group(1).strip()
            else:
                continue
                
        tokens = item.split()
        if tokens:
            law_part = tokens[0]
            if _RE_VALID_LAW_SUFFIX.search(law_part) or _RE_ARTICLE_START.match(item):
                cleaned_items.append(item)
                
    final_seen = set()
    final_list = []
    for ci in cleaned_items:
        if ci not in final_seen:
            final_seen.add(ci)
            final_list.append(ci)
            
    return ", ".join(final_list)


# ── 표준 데이터 전처리 ──────────────────────────────────────────────

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    for pat in _CLEAN_PATTERNS:
        text = pat.sub("", text)
    text = _RE_WHITESPACE.sub(" ", text).strip()
    return text


def convert_hanja(text: str) -> str:
    if not (_hanja2hangul and text):
        return text
    try:
        return _hanja2hangul(text)
    except Exception:
        return text


def format_date(date_str: str) -> str:
    s = str(date_str).strip()
    if re.match(r"^\d{8}$", s):
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return s


def validate_label(record: dict, idx: int, key: str) -> bool:
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

    csv_paths = [p for p in base.rglob("*") if p.is_file() and p.suffix.lower() == ".csv"]
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


# ── 메인 실행부 ──────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("형사법 LLM 데이터 파이프라인 가동 (함수 모듈화 릴리즈)")
    print("=" * 60)

    source_indexes: dict[str, dict] = {}
    for dt in DOCU_TYPES:
        source_indexes[dt] = load_source_csvs(dt)
        print(f"  {dt}: {len(source_indexes[dt]):,}개 문서 메모리 매핑 완료")

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

    print(f"\n[3/3] JSONL 저장 ➔ {OUTPUT_FILE}")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for row in tqdm(results, desc="  JSONL 저장", unit="건"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print("처리 완료 최종 통계")
    print("=" * 60)
    total_saved    = len(results)
    total_skip     = sum(v["skip"] for v in stats.values())
    total_fallback = sum(v["fallback"] for v in stats.values())
    total_input    = sum(v["total"] for v in stats.values())

    for key, s in sorted(stats.items()):
        saved = s["total"] - s["skip"]
        print(f"  {key:20s}: 입력 {s['total']:>6,}  저장 {saved:>6,}  정답대체 {s['fallback']:>5,}  스킵 {s['skip']:>4,}")

    print("-" * 60)
    print(f"  총 입력 문서 수: {total_input:,}건")
    print(f"  총 대체(정답대입): {total_fallback:,}건 💡")
    print(f"  최종 정제 완료  : {total_saved:,}건")
    print(f"  총 스킵 처리 수 : {total_skip:,}건")
    print(f"  출력 위치       : {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()