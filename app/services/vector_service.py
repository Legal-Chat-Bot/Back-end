import app.db.vector.client as pinecone
from app.db.vector.embedding import embed_query as embed_full


def embed_query(text: str):
    return embed_full(text)


def search_pinecone(
    question: str,
    context_mode: str,     # general / document / hybrid
    user_id: str,
    top_k: int = 5,
) -> list[dict]:
    emb = embed_full(question)

    def to_result(match, source_type: str) -> dict:
        return {
            "score": match["score"],
            "source_type": source_type,
            "metadata": match["metadata"],
        }

    if context_mode == "general":
        public_matches = pinecone.query(
            dense_vector=emb.dense,
            sparse_vector=emb.sparse,
            namespace=pinecone.public_namespace(),
            top_k=top_k,
        )
        results = [to_result(m, "legal_vector") for m in public_matches]
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    if context_mode == "document":
        private_matches = pinecone.query(
            dense_vector=emb.dense,
            sparse_vector=emb.sparse,
            namespace=pinecone.user_namespace(user_id),
            top_k=top_k,
        )
        results = [to_result(m, "document") for m in private_matches]
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    if context_mode == "hybrid":
        document_k = max(1, round(top_k * 0.6))  # top_k=5면 3
        public_k = top_k - document_k            # top_k=5면 2

        private_matches = pinecone.query(
            dense_vector=emb.dense,
            sparse_vector=emb.sparse,
            namespace=pinecone.user_namespace(user_id),
            top_k=top_k,
        )

        public_matches = pinecone.query(
            dense_vector=emb.dense,
            sparse_vector=emb.sparse,
            namespace=pinecone.public_namespace(),
            top_k=top_k,
        )

        private_results = [to_result(m, "document") for m in private_matches]
        public_results = [to_result(m, "legal_vector") for m in public_matches]

        private_results.sort(key=lambda x: x["score"], reverse=True)
        public_results.sort(key=lambda x: x["score"], reverse=True)

        results = private_results[:document_k] + public_results[:public_k]

        if len(results) < top_k:
            remaining = private_results[document_k:] + public_results[public_k:]
            remaining.sort(key=lambda x: x["score"], reverse=True)
            results.extend(remaining[:top_k - len(results)])

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    raise ValueError(f"Invalid context_mode: {context_mode}")


def is_legal_domain(search_results: list[dict], threshold: float = 0.6) -> bool:
    if not search_results:
        return False

    top_score = search_results[0]["score"]
    return top_score >= threshold


def is_legal_file(extracted_text: str, threshold: float = 0.6) -> bool:
    results = search_pinecone(
        question=extracted_text[:500],
        context_mode="general",
        user_id="",
        top_k=3,
    )
    return is_legal_domain(results, threshold)