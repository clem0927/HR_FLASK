# src/young/chatbot/tfidf_index.py

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .text_utils import normalize_korean_text

def build_tfidf_index(docs: list[dict]):
    local_corpus: list[str] = []
    for doc in docs:
        local_corpus.append(
            f"{doc['title']}\n{doc['content']}".strip()
        )

    local_vectorizer = TfidfVectorizer(
        preprocessor=normalize_korean_text,
        ngram_range=(1, 2)
    )
    local_doc_vectors = local_vectorizer.fit_transform(local_corpus)

    return local_corpus, local_vectorizer, local_doc_vectors


def retrieve_top_docs(
        query: str,
        documents: list[dict],
        vectorizer,
        doc_vectors,
        top_k: int = 3
):
    if not vectorizer or doc_vectors is None:
        return []

    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, doc_vectors)[0]
    ranked_indices = np.argsort(-sims)

    return [
        (documents[idx], float(sims[idx]))
        for idx in ranked_indices[:top_k]
    ]


def format_context(docs_with_scores, min_score: float = 0.0) -> str:
    if not docs_with_scores:
        return "관련된 인사관리 문서를 찾지 못했다."

    lines: list[str] = []
    for i, (doc, score) in enumerate(docs_with_scores, start=1):
        if score < min_score:
            continue

        lines.append(f"[문서 {i}] 질문: {doc['title']}")
        lines.append(f"답변: {doc['content']}")
        lines.append("")

    return "\n".join(lines) if lines else "관련된 인사관리 문서를 찾지 못했다."
