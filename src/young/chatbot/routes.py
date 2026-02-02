# src/young/chatbot/routes.py

from flask import Blueprint, request, jsonify
from .question_service import handle_question  # handle_question만 가져오기
from ..db import get_connection

bp = Blueprint("young_chatbot", __name__, url_prefix="/chatbot")

@bp.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "메시지가 비어 있습니다."}), 400

    # 비즈니스 로직(검색, 로그, LLM)은 service에서 한 번에 처리
    result = handle_question(user_message)

    # 에러 응답 처리
    if result.get("answer") == "서버 내부 오류가 발생했습니다.":
        return jsonify(result), 500

    return jsonify(result)

@bp.route("/stats", methods=["GET"])
def get_stats():
    conn = get_connection()
    cursor = conn.cursor()

    # 카운트가 높은 순서대로 상위 5개 이슈 가져오기
    cursor.execute("""
                   SELECT ISSUE_TITLE, ISSUE_COUNT, UPDATED_AT
                   FROM ISSUE_CLUSTER
                   ORDER BY ISSUE_COUNT DESC
                   """)

    rows = cursor.fetchall()
    stats = [
        {"title": row[0], "count": row[1], "updated_at": row[2].strftime('%Y-%m-%d %H:%M')}
        for row in rows
    ]

    cursor.close()
    conn.close()
    return jsonify(stats)