import asyncio
import re

# 네가 만든 모듈들 (실제 구조에 맞춤)
from app.services.vector_service import embed_query, search_pinecone, is_legal_domain
from app.services.prompt_service import assemble_messages
from app.services.service import generate_answer

# 조/항 추출 정규식
RE_JO = re.compile(r"제\d+조(?:의\d+)?")   # 제30조, 제30조의2
RE_HANG = re.compile(r"제\d+항")            # 제1항

TEST_QUESTIONS = [
    "뺑소니의 정의와 도망갔을 때에 형량",
    "보험사기에 대해서",
]


def extract_refs(text: str):
    """텍스트에서 조/항 뽑기"""
    return RE_JO.findall(text), RE_HANG.findall(text)


async def run_pipeline(question: str):
    print(f"\n{'='*70}")
    print(f"[질문] {question}")
    print('='*70)

    # 1. 임베딩
    qvec = embed_query(question)
    print(f"\n[1] 임베딩 완료 → {len(qvec)}차원")

    # 2. Pinecone 검색 (Top-K) — 네 코드는 검색 먼저 함
    results = search_pinecone(
        query_vector=qvec,
        context_mode="general",
        user_id="",
        top_k=3,
    )
    print(f"[2] Top-K 검색 완료 → {len(results)}개")

    # 3. 거름망 — 검색결과를 받아서 판별 (네 코드 구조)
    is_legal = is_legal_domain(results)
    top_score = results[0]["score"] if results else 0
    print(f"[3] 거름망 → {'법률 질문 ✅' if is_legal else '비법률 ❌'} (최고점수: {top_score:.4f})")
    if not is_legal:
        print("    → 비법률 질문이라 중단")
        return

    # 3-1. Top-K 본문에 조/항 있나 확인
    print(f"\n{'-'*70}")
    print("[Top-K 본문 조/항 분석]")
    topk_all_text = ""
    for idx, r in enumerate(results, 1):
        text = r.get("metadata", {}).get("text", "") or r.get("metadata", {}).get("qa_text", "")
        topk_all_text += " " + text
        jo, hang = extract_refs(text)
        preview = text[:40].replace("\n", " ")
        print(f"  판례{idx}: 조 {len(jo)}개 {jo} | 항 {len(hang)}개 {hang}")
        print(f"          (본문: {preview}...)")
    topk_jo, topk_hang = extract_refs(topk_all_text)
    print(f"  → Top-K 전체 조: {sorted(set(topk_jo))}")
    print(f"  → Top-K 전체 항: {sorted(set(topk_hang))}")

    # 4. 프롬프트 조립 — SYSTEM_PROMPT는 내부에서 붙음
    messages = assemble_messages(question, results, [])
    print(f"\n[4] 프롬프트 조립 완료 → {len(messages)}개 메시지 (system + user)")

    # 5. LLM 답변 생성
    print(f"[5] 답변 생성 중... (CPU 추론, 1~3분)")
    answer = await generate_answer(messages)

    # 5-1. 답변의 조/항 분석
    ans_jo, ans_hang = extract_refs(answer)
    print(f"\n[답변]\n{answer}")
    print(f"\n{'-'*70}")
    print(f"[답변 조/항 분석]")
    print(f"  답변 조: {ans_jo}")
    print(f"  답변 항: {ans_hang}")

    # 6. 핵심: 답변 조항이 Top-K에 있나? (환각 검증 시뮬레이션)
    print(f"\n{'-'*70}")
    print("[★ 환각 검증 시뮬레이션]")
    if not ans_jo:
        print("  답변에 조문 없음 → 검증할 것 없음")
    else:
        topk_jo_set = set(topk_jo)
        for jo in sorted(set(ans_jo)):
            if jo in topk_jo_set:
                print(f"  ✅ '{jo}' → Top-K에 있음 (확인됨)")
            else:
                print(f"  ⚠️ '{jo}' → Top-K에 없음 (환각 의심)")


async def main():
    for q in TEST_QUESTIONS:
        await run_pipeline(q)


if __name__ == "__main__":
    asyncio.run(main())