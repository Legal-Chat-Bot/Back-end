# 법률 데이터셋 수집 파이프라인
# 데이터 로딩 → 정제 → 청킹 확인 → JSON 저장
import json
import re
import unicodedata
from pathlib import Path
from datasets import load_dataset
from kss import Kss
import sys
# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.db.vector.chunker import Chunker, ChunkConfig

# ── 설정 ──────────────────────────────────────────────────
OUTPUT_FILE  = Path(__file__).parent / "legal.json"
DATASET_NAME = "LDKSolutions/KR-legal-60K-dataset-jsonl"

# ── KSS 초기화 (한 번만) ─────────────────────────────────
_hanja2hangul = Kss("hanja2hangul")

# ── 청커 초기화 ───────────────────────────────────────────
_chunker = Chunker(ChunkConfig(
    max_chars=500,
    overlap_chars=50,
    min_chars=20,
))


# ── 텍스트 정제 ───────────────────────────────────────────
def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _hanja2hangul(text)
    text = re.sub(r"\[별표\s*\d*\]", "", text)
    text = re.sub(r"\[별지\s*제?\d*호?\s*서식?\]", "", text)
    text = re.sub(r"\[시행\s*[\d\.\s]+\]", "", text)
    text = re.sub(r"\[법률\s*제\d+호\]", "", text)
    text = re.sub(r"<표\s*\d*>", "", text)
    text = re.sub(r"<그림\s*\d*>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── date 포맷 변환 ────────────────────────────────────────
def format_date(date_str: str) -> str:
    """20221027 → 2022-10-27"""
    date_str = str(date_str).strip()
    if len(date_str) == 8:
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    return date_str


# ── 검증 ──────────────────────────────────────────────────
def validate(idx: int, cleaned: str, category: str, law_date: str) -> bool:
    errors = []
    if len(cleaned) < 10:
        errors.append(f"text 너무 짧음 ({len(cleaned)}자)")
    if not category:
        errors.append("category 없음")
    if law_date and not re.match(r"\d{4}-\d{2}-\d{2}", law_date):
        errors.append(f"date 형식 오류: {law_date}")
    if errors:
        print(f"  ⚠ [{idx}] 검증 실패: {', '.join(errors)}")
        return False
    return True


# ── 메인 ──────────────────────────────────────────────────
def main():
    print("⏳ 데이터셋 로딩 중...")

    try:
        dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return

    results = []
    skip_count = 0

    for idx, sample in enumerate(dataset):
        raw_text = sample.get("text", "")
        category = sample.get("type", "")
        law_date = format_date(sample.get("date", ""))

        if not raw_text:
            print(f"  ⚠ [{idx}] text 없음, 스킵")
            skip_count += 1
            continue

        # 정제
        cleaned = clean_text(raw_text)

        # 검증
        if not validate(idx, cleaned, category, law_date):
            skip_count += 1
            continue

        # 청킹
        chunks = _chunker.chunk(cleaned)

        results.append({
            "text":         cleaned,
            "category":     category,
            "law_date":     law_date,
            "total_chunks": len(chunks),
            "chunks": [
                {
                    "chunk_index": c.chunk_index,
                    "text":        c.text,
                    "char_count":  c.char_count,
                }
                for c in chunks
            ],
        })

        if idx % 1000 == 0:
            print(f"  ✅ {idx}번째 처리 중... (저장 {len(results)}개 / 스킵 {skip_count}개)")

    # JSON 저장
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 완료! 총 {len(results)}개 저장 → {OUTPUT_FILE}")
    print(f"   스킵: {skip_count}개")


if __name__ == "__main__":
    main()
