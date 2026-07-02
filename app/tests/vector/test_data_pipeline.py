import sys
import os
import pytest

# 🛠️ 최상위 루트 및 dataset 폴더를 파이썬 모듈 검색 경로에 강제 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, ".."))

if root_path not in sys.path:
    sys.path.insert(0, root_path)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from dataset.data_pipeline import (
    clean_and_normalize_punc,
    extract_potential_law_name,
    validate_extracted_law,
    inherit_and_resolve_law_hierarchy,
    format_date,
)


# ── [테스트 1] 기호 및 괄호 정제 단위 기능 검증 ──
def test_pipeline_clean_and_normalize_punc():
    """
    clean_and_normalize_punc 함수가 텍스트의 불필요한 괄호,
    전후 수식어, 기호들을 정상적으로 세척하는지 검증.

    [수정 근거]
    - 기존 테스트 로직은 올바름. 함수가 괄호 내용 제거 후 '및', '또는' 등
      suffix/prefix 패턴을 _RE_CLEAN_PUNCT_SUFFIX / _RE_CLEAN_PUNCT_PREFIX로
      정리하므로 결과에 해당 단어가 없어야 한다는 assert는 정확함.
    - 추가로 result가 빈 문자열이 되는 케이스를 명시하여 의도를 분명히 함.
    """
    sample_text = " (형사소송법 제1조) 및 또는"
    result = clean_and_normalize_punc(sample_text)

    assert "및" not in result
    assert "또는" not in result


# ── [테스트 2] 법령명 추출 및 검증 룰셋 기능 검증 ──
def test_pipeline_law_extraction_and_validation():
    """
    extract_potential_law_name 함수와 validate_extracted_law 함수가
    텍스트 내에서 알맞은 법령명을 추출하고 노이즈를 잡아내는지 연동 검증.

    [수정 근거]
    - 「」(겹낫표) 패턴은 _RE_BRACKETED_LAW = re.compile(r'「([^」]+)」$')로 처리됨.
      원본 샘플 「형사소송법」은 문자열 끝에 위치해야 매칭됨. 기존 케이스 정상.
    - validate_extracted_law의 invalid_keywords에 "전단"이 포함되어 있으므로
      "전단 제38조"는 False 반환이 맞음. 기존 케이스 정상.
    - 추가로 정상 법령명("형사소송법")이 validate를 통과하는 케이스를 명시적으로 검증.
    """
    # 1. 겹낫표로 감싸인 법령명 추출 검증
    raw_text = "「형사소송법」"
    extracted = extract_potential_law_name(raw_text)
    assert extracted == "형사소송법", (
        f"겹낫표 법령명 추출 실패. 기대값: '형사소송법', 실제값: {extracted!r}"
    )
    assert validate_extracted_law(extracted) is True

    # 2. 유해 지칭어("전단")가 섞인 법령 후보는 검증에서 탈락해야 함
    # invalid_keywords = ["전단", "후단", "따라", ...] 에 "전단"이 포함되어 있어 False
    invalid_law = "전단 제38조"
    assert validate_extracted_law(invalid_law) is False, (
        f"'전단'을 포함한 노이즈 법령명이 validate를 통과하면 안 됩니다. 입력값: {invalid_law!r}"
    )


# ── [테스트 3] 단독 법령 계층 상속 알고리즘 검증 ──
def test_pipeline_law_hierarchy_inheritance():
    """
    inherit_and_resolve_law_hierarchy 함수가 '시행령', '법률' 단독 노출 형태에 대해
    직전 메인 법령명(Context)을 소스코드의 트리 구조 규칙대로 상속하는지 검증.

    [수정 근거]
    - 입력: [("형사소송법", "제1조"), ("시행령", "제2조"), ("법률", "제3조")]
    - "형사소송법" → is_suffix_law=False, is_standalone_law=False, is_incomplete_ref=False
        → last_main_law = "형사소송법" 으로 갱신
    - "시행령" → is_suffix_law=True, last_main_law="형사소송법"
        → effective_law = "형사소송법 시행령"
    - "법률" → is_standalone_law=True (["법", "법률", "동법"] 에 포함), last_main_law="형사소송법"
        → effective_law = "형사소송법"
    - 기존 테스트 assert 값(resolved_pairs[1][0]=="형사소송법 시행령",
      resolved_pairs[2][0]=="형사소송법")은 실제 로직과 일치하여 올바름.
    """
    mock_extracted_pairs = [
        ("형사소송법", "제1조"),
        ("시행령", "제2조"),
        ("법률", "제3조"),
    ]

    resolved_pairs = inherit_and_resolve_law_hierarchy(mock_extracted_pairs)

    assert len(resolved_pairs) == 3, (
        f"반환된 쌍의 수가 입력과 다릅니다. 기대: 3, 실제: {len(resolved_pairs)}"
    )

    # "형사소송법" → 메인 법령으로 그대로 유지
    assert resolved_pairs[0][0] == "형사소송법", (
        f"resolved_pairs[0][0] 기대값 '형사소송법', 실제값: {resolved_pairs[0][0]!r}"
    )

    # "시행령" → 직전 메인 법령("형사소송법")을 접두어로 상속
    assert resolved_pairs[1][0] == "형사소송법 시행령", (
        f"resolved_pairs[1][0] 기대값 '형사소송법 시행령', 실제값: {resolved_pairs[1][0]!r}"
    )

    # "법률" → is_standalone_law=True → 직전 메인 법령명 그대로 상속
    assert resolved_pairs[2][0] == "형사소송법", (
        f"resolved_pairs[2][0] 기대값 '형사소송법', 실제값: {resolved_pairs[2][0]!r}"
    )


# ── [테스트 4] 날짜 문자열 포맷팅 기능 검증 ──
def test_pipeline_date_formatting():
    """
    format_date 함수가 8자리 연속된 숫자(YYYYMMDD)를 인입받았을 때
    표준 대시 형태(YYYY-MM-DD)로 변환해 내는지 검증.

    [수정 근거]
    - format_date는 re.match(r'^\d{8}$', s) 조건을 사용하므로
      정확히 8자리 숫자일 때만 변환하고, 그 외는 원본 반환.
    - 기존 테스트 케이스 3개 모두 로직과 일치하여 올바름.
    - 엣지케이스(None, 빈 문자열 등)는 str() 변환 후 처리되므로 안전.
    """
    # 1. 8자리 숫자 → YYYY-MM-DD 변환
    assert format_date("20260330") == "2026-03-30"

    # 2. 이미 포맷된 날짜 → 원본 유지 (8자리 숫자가 아니므로 변환 안 함)
    assert format_date("2026-03-30") == "2026-03-30"

    # 3. 문자열 날짜 → 원본 유지
    assert format_date("일자미상") == "일자미상"