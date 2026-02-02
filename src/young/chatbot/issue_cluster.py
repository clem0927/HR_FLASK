import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ..db import get_connection

# Sentence Transformer 모델
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
# 설정값
SIMILARITY_THRESHOLD = 0.70

def _embed(text: str) -> np.ndarray:
    return model.encode([text])[0]


# src/young/chatbot/issue_cluster.py 파일의 해당 부분 수정

def _load_all_clusters():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
                   SELECT ISSUE_ID, ISSUE_NORM, EMBEDDING
                   FROM ISSUE_CLUSTER
                   """)

    rows = cursor.fetchall()
    clusters = []

    for issue_id, norm, embedding_lob in rows:
        # 핵심: embedding_lob이 Oracle LOB 객체일 경우 .read()로 내용을 가져와야 함
        # 데이터가 문자열(str)인 경우와 LOB인 경우를 모두 대응하게 처리
        if hasattr(embedding_lob, "read"):
            embedding_json = embedding_lob.read()
        else:
            embedding_json = embedding_lob

        clusters.append({
            "issue_id": issue_id,
            "norm": norm,
            "embedding": np.array(json.loads(embedding_json))
        })

    cursor.close()
    conn.close()

    return clusters

def upsert_issue_cluster(raw_text: str, normalized_text: str) -> int:
    query_vec = _embed(normalized_text)
    clusters = _load_all_clusters()

    best_issue_id = None
    best_score = 0.0

    for c in clusters:
        score = cosine_similarity([query_vec], [c["embedding"]])[0][0]

        #  이 로그를 추가해서 점수를 꼭 확인하세요!
        print(f"[DEBUG] DB질문: {c['norm']} vs 새질문: {normalized_text} | 점수: {score:.4f}")

        if score > best_score:
            best_score = score
            best_issue_id = c["issue_id"]

    conn = get_connection()
    cursor = conn.cursor()

    # 기존 이슈 업데이트 (점수가 문턱값보다 높을 때)
    if best_score >= SIMILARITY_THRESHOLD:
        print(f"[INFO] 유사 이슈 발견(ID:{best_issue_id}, Score:{best_score:.4f}). 카운트 증가.")
        cursor.execute("""
                       UPDATE ISSUE_CLUSTER
                       SET ISSUE_COUNT = ISSUE_COUNT + 1,
                           UPDATED_AT = SYSDATE
                       WHERE ISSUE_ID = :id
                       """, id=best_issue_id)

        conn.commit()
        cursor.close()
        conn.close()

        return best_issue_id

    # 신규 이슈 생성
    cursor.execute("""
                   INSERT INTO ISSUE_CLUSTER (
                       ISSUE_TITLE,
                       ISSUE_NORM,
                       EMBEDDING,
                       ISSUE_COUNT
                   ) VALUES (
                                :title,
                                :norm,
                                :embedding,
                                1
                            )
                   """, {
                       "title": raw_text,
                       "norm": normalized_text,
                       "embedding": json.dumps(query_vec.tolist())
                   })

    conn.commit()

    cursor.execute("SELECT ISSUE_CLUSTER_SEQ.CURRVAL FROM dual")
    issue_id = cursor.fetchone()[0]

    cursor.close()
    conn.close()

    return issue_id
