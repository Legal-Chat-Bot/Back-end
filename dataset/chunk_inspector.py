# 청킹 결과 확인 도구
# legal.json 로드 → 청크 내용/통계 출력

import json
from pathlib import Path
import sys
# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.db.vector.chunker import Chunker, ChunkConfig
LEGAL_JSON = Path(__file__).parent / "legal.json"

_chunker = Chunker(ChunkConfig(max_chars=500, overlap_chars=50, min_chars=20))


def inspect_from_json(max_docs: int = 5):
    """저장된 legal.json에서 청킹 결과 확인"""
    if not LEGAL_JSON.exists():
        print(f"❌ {LEGAL_JSON} 없음. pipeline.py 먼저 실행하세요.")
        return

    with open(LEGAL_JSON, encoding="utf-8") as f:
        data = json.load(f)

    print(f"📂 총 문서 수: {len(data)}개\n")
    print("=" * 70)

    for i, doc in enumerate(data[:max_docs]):
        print(f"\n📄 문서 #{i+1} | 카테고리: {doc['category']} | 날짜: {doc['law_date']}")
        print(f"   전체 글자 수: {len(doc['text'])}자 | 청크 수: {doc['total_chunks']}개")
        print("-" * 70)

        for chunk in doc["chunks"]:
            print(f"  [청크 {chunk['chunk_index']}] ({chunk['char_count']}자)")
            # 100자까지만 미리보기
            preview = chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"]
            print(f"    {preview}")

        print("=" * 70)


def inspect_from_text(text: str):
    """텍스트 직접 입력해서 청킹 결과 확인"""
    stats = _chunker.chunk_with_stats(text)

    print(f"\n📊 청킹 통계")
    print(f"  총 청크 수 : {stats['total_chunks']}")
    print(f"  총 글자 수 : {stats['total_chars']}")
    print(f"  평균 글자 수: {stats['avg_chars']:.1f}")
    print(f"  최소 글자 수: {stats['min_chars']}")
    print(f"  최대 글자 수: {stats['max_chars']}")
    print("-" * 70)

    for chunk in stats["chunks"]:
        print(f"\n[청크 {chunk.chunk_index}] ({chunk.char_count}자)")
        print(f"  {chunk.text}")


if __name__ == "__main__":
    # ── 모드 1: JSON 파일에서 확인 ──
    inspect_from_json(max_docs=3)

    # ── 모드 2: 텍스트 직접 테스트 ──
    sample = """
    第一條 이 法은 國民의 基本權을 保障함을 目的으로 한다.
    第二條 모든 國民은 法앞에 平等하다. 누구든지 性別·宗敎 또는 社會的 身分에 의하여
    政治的·經濟的·社會的·文化的 생활의 모든 領域에 있어서 差別을 받지 아니한다.
    第三條 行政機關은 法律에 따라 그 職務를 遂行하여야 한다.
    """
    print("\n\n── 직접 텍스트 테스트 ──")
    inspect_from_text(sample)
