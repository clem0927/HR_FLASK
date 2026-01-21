from flask import Blueprint, request, jsonify
import requests

# ==============================
# Blueprint 생성
# ==============================
bp = Blueprint(
    "young_phase",
    __name__,
    url_prefix="/phase"
)

# ==============================
# Ollama 설정
# ==============================
MODEL_NAME = "gemma3:4b"
OLLAMA_URL = "http://localhost:11434/api/chat"

# ==============================
# 폭포수 모델 단계별 기본 가이드
# (TF-IDF CSV 대신 기초 지식 베이스 역할)
# ==============================
WATERFALL_PHASE_GUIDE = {
    "요구사항분석": (
        "사용자 및 이해관계자의 요구사항을 수집하고 문서화한다. "
        "기능 요구사항, 비기능 요구사항, 제약사항을 명확히 정의한다."
    ),
    "설계": (
        "요구사항을 기반으로 시스템 구조를 설계한다. "
        "아키텍처 설계, 데이터베이스 설계, 인터페이스 설계를 포함한다."
    ),
    "구현": (
        "설계 문서를 기반으로 실제 소프트웨어를 개발한다. "
        "코딩 표준을 준수하고 각 기능을 구현한다."
    ),
    "테스트": (
        "구현된 기능이 요구사항을 충족하는지 검증한다. "
        "단위 테스트, 통합 테스트, 시스템 테스트를 수행한다."
    ),
    "유지보수": (
        "운영 중 발생하는 오류를 수정하고 기능을 개선한다. "
        "환경 변화에 대응하여 시스템을 지속적으로 관리한다."
    )
}

# ==============================
# Ollama 호출 함수
# ==============================
def ask_phase_ai(phase_name: str, methodology: str) -> str:
    base_guide = WATERFALL_PHASE_GUIDE.get(phase_name)

    if not base_guide:
        return "해당 단계에 대한 기본 설명이 정의되어 있지 않습니다."

    system_message = {
        "role": "system",
        "content": (
            f"너는 IT 회사에서 사용하는 {methodology} 모델 기반 "
            "소프트웨어 프로젝트 관리 도우미다."
        )
    }

    user_message = {
        "role": "user",
        "content": (
            f"{methodology} 모델의 '{phase_name}' 단계에 대해 "
            "실제 프로젝트에서 바로 사용할 수 있는 단계 설명을 작성해라.\n\n"
            f"기초 조건:\n{base_guide}\n\n"
            "- 실무 문서에 바로 붙여 넣을 수 있게 작성\n"
            "- 교과서 말투는 피할 것\n"
            "- 3~5문장 분량"
        )
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [system_message, user_message],
        "stream": False
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    return data.get("message", {}).get("content", "").strip()

# ==============================
#  AI 단계 설명 생성 API
# ==============================
@bp.route("/ai-description", methods=["POST"])
def generate_phase_description():
    data = request.get_json(silent=True) or {}

    phase_name = data.get("phaseName")
    methodology = data.get("methodology", "폭포수")

    if not phase_name:
        return jsonify({"error": "phaseName이 필요합니다."}), 400

    try:
        description = ask_phase_ai(phase_name, methodology)
        return jsonify({"description": description})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
