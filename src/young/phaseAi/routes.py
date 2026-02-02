from flask import Blueprint, request, jsonify
import requests

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
# 폭포수 단계 기본 가이드
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
# Ollama 호출 공통 함수
# ==============================
def call_ollama(system_prompt: str, user_prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    return data.get("message", {}).get("content", "").strip()

# ==============================
# AI 단계 설명 / 목표 생성
# ==============================
@bp.route("/ai-description", methods=["POST"])
def generate_phase_ai():
    data = request.get_json(silent=True) or {}

    phase_name = data.get("phaseName")
    methodology = data.get("methodology", "폭포수")
    project_name = data.get("projectName", "")
    req_type = data.get("type", "description")  # description | goal

    if not phase_name:
        return jsonify({"error": "phaseName이 필요합니다."}), 400

    base_guide = WATERFALL_PHASE_GUIDE.get(phase_name, "")

    try:
        # ==========================
        # 단계 설명 (기법 중심)
        # ==========================
        if req_type == "description":
            system_prompt = (
                f"너는 {methodology} 개발 방법론에 정통한 "
                "소프트웨어 프로젝트 관리 전문가다."
            )

            user_prompt = (
                f"{methodology} 모델의 '{phase_name}' 단계에 대해 설명하라.\n\n"
                f"기초 참고 내용:\n{base_guide}\n\n"
                "- 실무 문서에 바로 붙여 넣을 수 있도록 작성\n"
                "- 교과서식 설명은 피할 것\n"
                "- 단계의 목적과 핵심 활동 위주\n"
                "- 3~5문장 분량"
            )

        # ==========================
        # 단계 목표 (프로젝트 중심)
        # ==========================
        elif req_type == "goal":
            system_prompt = (
                "너는 IT 회사의 숙련된 PM(Project Manager)이다."
            )

            user_prompt = (
                f"'{project_name}' 프로젝트에서 "
                f"'{phase_name}' 단계의 목표를 작성하라.\n\n"
                "- PM이 팀원에게 공유하는 문체\n"
                "- 실제 업무 지시처럼 구체적으로 작성\n"
                "- 산출물, 인터뷰, 검토 대상 등을 포함\n"
                "- 3~5문장 분량"
            )

        else:
            return jsonify({"error": "type 값이 올바르지 않습니다."}), 400

        result = call_ollama(system_prompt, user_prompt)
        return jsonify({"description": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
