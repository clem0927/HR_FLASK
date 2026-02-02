# src/young/chatbot/routes.py

import os
import csv
import requests
import numpy as np
from flask import Blueprint, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .text_utils import normalize_korean_text
# ==============================
# Blueprint 생성 (요청한 형태 그대로)
# ==============================
bp = Blueprint(
    "young_chatbot",
    __name__,
    url_prefix="/chatbot"
)

# ==============================
# Ollama 모델 설정
# ==============================
model_name = "gemma3:4b"
api_url = "http://localhost:11434/api/chat"

# ==============================
# CSV 디렉터리 경로 (__file__ 기준 절대경로)
# src/data/chatbot
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "../../data/chatbot")
)

# ==============================
# 전역 변수
# ==============================
chat_history: list[dict] = []
documents: list[dict] = []
corpus: list[str] = []
vectorizer: TfidfVectorizer | None = None
doc_vectors = None


# ==============================
# CSV 디렉터리 내 모든 CSV 로드
# ==============================
def load_documents_from_csv_dir(dir_path: str) -> list[dict]:
    docs: list[dict] = []

    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"CSV 디렉터리를 찾을 수 없습니다: {dir_path}")

    for filename in os.listdir(dir_path):
        if not filename.lower().endswith(".csv"):
            continue

        full_path = os.path.join(dir_path, filename)

        with open(full_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                question = (row.get("text") or "").strip()
                answer = (row.get("intent") or "").strip()

                if not question or not answer:
                    continue

                docs.append({
                    "title": question,
                    "content": answer
                })

    if not docs:
        raise ValueError("CSV 디렉터리에서 유효한 문서를 하나도 읽지 못했습니다.")

    return docs

# ==============================
# TF-IDF 인덱스 생성
# ==============================
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

# ==============================
# 서버 시작 시 CSV 로드 & 벡터화
# ==============================
try:
    documents = load_documents_from_csv_dir(CSV_DIR)
    corpus, vectorizer, doc_vectors = build_tfidf_index(documents)
    print(f"[INFO] HR CSV 문서 {len(documents)}개 로드 완료")
    print(f"[INFO] CSV_DIR = {CSV_DIR}")
except Exception as e:
    print(f"[WARN] HR 지식 베이스 초기화 실패: {e}")
    documents = []
    corpus = []
    vectorizer = None
    doc_vectors = None

# ==============================
# 벡터 검색
# ==============================
def retrieve_top_docs(query: str, top_k: int = 3):
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

def build_context_text(user_message: str) -> str:
    top_docs = retrieve_top_docs(user_message, top_k=5)
    return format_context(top_docs)

# ==============================
# Ollama API 호출
# ==============================
def ask_ollama(user_message: str, context_text: str) -> str:
    system_role_message = {
        "role": "system",
        "content": (
            "너는 회사 내부 인사관리(HR) 챗봇이다. "
            "급여, 휴가, 근태, 복지, 인사 규정 관련 질문에 답변한다."
        ),
    }

    context_message = {
        "role": "system",
        "content": (
            "다음은 HR CSV에서 검색된 관련 문서이다.\n"
            f"{context_text}\n\n"
            "위 문서 내용에서만 정보를 사용해 한국어로 답변하라. "
            "문서에 없는 내용이면 "
            "'해당 질문은 인사관리 규정에 없는 내용이라 답변할 수 없습니다.'라고 답하라."
        ),
    }

    messages: list[dict] = [system_role_message, context_message]

    if chat_history:
        messages.extend(chat_history[-8:])

    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False
    }

    resp = requests.post(api_url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    return data.get("message", {}).get("content", "").strip()

# ==============================
# 챗봇 API 라우트
# ==============================
@bp.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "메시지가 비어 있습니다."}), 400

    if not documents:
        return jsonify({"error": "HR 지식 베이스가 초기화되지 않았습니다."}), 500

    try:
        save_question_log(
            raw_question=user_message,
            normalized_question=normalize_korean_text(user_message)
        )

        context_text = build_context_text(user_message)
        answer = ask_ollama(user_message, context_text)

        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": answer})

        if len(chat_history) > 20:
            del chat_history[:-20]

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
