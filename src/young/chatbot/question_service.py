# src/young/chatbot/question_service.py

from typing import Dict
import traceback
import os

from .text_utils import normalize_korean_text
from .tfidf_index import retrieve_top_docs, format_context
from .ollama_client import ask_ollama
from .csv_loader import load_documents_from_csv_dir
from ..db import get_connection
from .issue_cluster import upsert_issue_cluster

# ==============================
# CSV 디렉터리 경로
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "../../data/chatbot")
)

# ==============================
# 전역 리소스
# ==============================
documents = []
vectorizer = None
doc_vectors = None
chat_history: list[dict] = []

# ==============================
# 서버 시작 시 CSV 로드 & 벡터화
# ==============================
try:
    from .tfidf_index import build_tfidf_index

    documents = load_documents_from_csv_dir(CSV_DIR)
    _, vectorizer, doc_vectors = build_tfidf_index(documents)

    print(f"[INFO] HR CSV 문서 {len(documents)}개 로드 완료")
except Exception as e:
    print("[WARN] HR 지식 베이스 초기화 실패")
    traceback.print_exc()
    documents = []
    vectorizer = None
    doc_vectors = None

# ==============================
# 질문 필터
# ==============================
BAD_WORDS = ["시발", "병신", "개새끼","개새", "욕설","좆됐다","존나","즐","조까","욕설"]

def is_invalid_question(raw_text: str, normalized_text: str) -> bool:
    if not raw_text:
        return True

    for bad in BAD_WORDS:
        if bad in raw_text:
            return True
        if bad in normalized_text:
            return True

    return False


# ==============================
# 질문 저장
# ==============================
def save_question_log(raw_question: str, normalized_question: str, issue_id: int):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO HR_CHAT_LOG (ID, QUESTION_RAW, QUESTION_NORM, ISSUE_ID)
            VALUES (HR_CHAT_LOG_SEQ.NEXTVAL, :q_raw, :q_norm, :issue_id)
            """,
            q_raw=raw_question,
            q_norm=normalized_question,
            issue_id=issue_id
        )

        conn.commit()
        print("[INFO] 질문 로그 저장 성공")

    except Exception:
        print("[ERROR] 질문 로그 저장 실패")
        traceback.print_exc()

    finally:
        cursor.close()
        conn.close()

# ==============================
# Context 생성
# ==============================
def build_context_text(query: str) -> str:
    return format_context(
        retrieve_top_docs(
            query=query,
            documents=documents,
            vectorizer=vectorizer,
            doc_vectors=doc_vectors,
            top_k=5
        )
    )

# ==============================
# 질문 처리 메인 엔트리
# ==============================
def handle_question(user_message: str) -> Dict:
    print(f"[INFO] 질문 수신: {user_message}")
    print(f"[DEBUG] RAW: {user_message}")

    try:
        for bad in BAD_WORDS:
            if bad in user_message:
                print("[WARN] RAW 욕설 차단")
                return {"answer": "업무와 관련된 질문만 가능합니다(욕설 금지!)"}
        # 1️⃣ 정규화
        normalized = normalize_korean_text(user_message)
        print(f"[DEBUG] 정규화 결과: {normalized}")

        # 2️⃣ 욕설 / 무효 질문 필터 (RAW + NORMALIZED)
        if is_invalid_question(user_message, normalized):
            print("[WARN] 욕설 또는 무효 질문 차단")
            return {"answer": "업무와 관련된 질문만 가능합니다(욕설 금지!)"}

        #  욕설 / 무효 질문 필터 통과 후
        issue_id = upsert_issue_cluster(
            raw_text=user_message,
            normalized_text=normalized
        )

        print(f"[INFO] 이슈 클러스터 ID: {issue_id}")

        # 3️⃣ 질문 로그 저장
        save_question_log(
            raw_question=user_message,
            normalized_question=normalized,
            issue_id=issue_id
        )

        # 4️⃣ 지식 베이스 체크
        if not documents or not vectorizer or doc_vectors is None:
            return {"answer": "HR 지식 베이스가 초기화되지 않았습니다."}

        # 5️⃣ 문서 검색 → Context 생성
        context_text = build_context_text(normalized)
        print("[DEBUG] Context 생성 완료")

        # 6️⃣ LLM 호출
        answer = ask_ollama(
            normalized,
            context_text,
            chat_history
        )
        print("[INFO] Ollama 응답 수신")

        # 7️⃣ 대화 히스토리 관리
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": answer})

        if len(chat_history) > 20:
            del chat_history[:-20]

        return {"answer": answer}

    except Exception:
        print("[FATAL] handle_question 처리 중 예외 발생")
        traceback.print_exc()
        return {"answer": "서버 내부 오류가 발생했습니다."}
