# src/gyu/routes.py
"""
포상 추천 AI API 엔드포인트
- /gyu/reward/health: 서버 상태 확인
- /gyu/reward/recommend: 포상 추천 요청
"""
from flask import Blueprint, request, jsonify

from .models.tfidf_recommender import get_recommender
from .config import RECOMMENDER_TYPE

bp = Blueprint(
    "gyu_reward",
    __name__,
    url_prefix="/gyu/reward"
)

# 추천 엔진 인스턴스 (lazy loading)
_recommender = None


def get_recommender_instance():
    """추천 엔진 싱글톤 인스턴스 반환"""
    global _recommender
    if _recommender is None:
        _recommender = get_recommender()
        print(f"[GYU] 추천 엔진 로드: {_recommender.get_model_type()}")
    return _recommender


@bp.route("/health", methods=["GET"])
def health_check():
    """
    서버 상태 확인 API

    Returns:
        {
            "status": "ok",
            "model_type": "TfidfRecommender",
            "version": "1.0.0"
        }
    """
    try:
        recommender = get_recommender_instance()
        return jsonify({
            "status": "ok",
            "model_type": recommender.get_model_type(),
            "config_type": RECOMMENDER_TYPE,
            "version": "1.0.0"
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@bp.route("/recommend", methods=["POST"])
def recommend():
    """
    포상 추천 API

    Request Body:
        {
            "employee": {
                "empId": "E001",
                "empName": "홍길동",
                "avgScore": 92.5,
                "comments": ["프로젝트 완료에 크게 기여함", "팀원들과 협업이 뛰어남"]
            },
            "policies": [
                {"policyId": 1, "policyName": "프로젝트 MVP", "description": "..."},
                {"policyId": 2, "policyName": "팀워크상", "description": "..."}
            ]
        }

    Returns:
        {
            "status": "success",
            "recommendations": [...],
            "extractedKeywords": [...],
            "overallSentiment": "positive"
        }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "status": "error",
                "error": "요청 데이터가 없습니다."
            }), 400

        # 요청 데이터 파싱
        employee = data.get("employee", {})
        policies = data.get("policies", [])

        emp_id = employee.get("empId", "unknown")
        emp_name = employee.get("empName", "unknown")
        comments = employee.get("comments", [])

        if not comments:
            return jsonify({
                "status": "error",
                "error": "평가 코멘트가 없습니다."
            }), 400

        if not policies:
            return jsonify({
                "status": "error",
                "error": "포상 정책이 없습니다."
            }), 400

        print(f"[GYU] 추천 요청 - 직원: {emp_name}({emp_id}), 코멘트 수: {len(comments)}, 정책 수: {len(policies)}")
        print(f"[GYU] 코멘트 내용: {comments}")
        print(f"[GYU] 정책 목록: {[p.get('policyName') for p in policies]}")

        # 추천 엔진 실행
        recommender = get_recommender_instance()
        result = recommender.analyze(comments, policies)

        print(f"[GYU] 추천 완료 - 추천 수: {len(result.get('recommendations', []))}")
        print(f"[GYU] 결과: {result}")

        return jsonify({
            "status": "success",
            **result
        }), 200

    except Exception as e:
        print(f"[GYU] 추천 오류: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@bp.route("/keywords", methods=["POST"])
def extract_keywords():
    """
    키워드 추출 API (디버깅/테스트용)

    Request Body:
        {
            "text": "분석할 텍스트..."
        }

    Returns:
        {
            "status": "success",
            "keywords": ["키워드1", "키워드2", ...]
        }
    """
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text:
            return jsonify({
                "status": "error",
                "error": "텍스트가 없습니다."
            }), 400

        recommender = get_recommender_instance()
        keywords = recommender.extract_keywords(text)

        return jsonify({
            "status": "success",
            "keywords": keywords
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500