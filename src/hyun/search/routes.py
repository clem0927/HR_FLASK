from flask import Flask, Blueprint, request, jsonify
from flask_cors import CORS  # 리액트 연동을 위해 필요
import requests
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# 1. Blueprint 설정
bp = Blueprint("hyun_search", __name__, url_prefix="/hyun/search")

# 2. Ollama 모델 설정
llm = ChatOllama(
    model="gemma3:4b",
    temperature=0,
)

# 3. 스프링부트 API 주소 (사원 정보 조회를 처리할 엔드포인트)
SPRING_BOOT_URL = "http://localhost:8080/api/emp/query"
@bp.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        user_question = data.get("question")
        print(f"1. 사용자 질문 수신: {user_question}")

        if not user_question:
            return jsonify({"status": "error", "message": "질문을 입력해주세요."}), 400

        # 4. 프롬프트 구성 (여기를 생략 없이 다 채워야 합니다)
        prompt = ChatPromptTemplate.from_template("""
        너는 Oracle SQL 전문가야. 다음 테이블 구조를 바탕으로 사용자의 질문을 SQL로 변환해줘.
        
        [Table: EMP_SKILL]
        - EMP_ID (사원번호)
        - SKILL_NAME (기술명, 예: 'Java', 'Python', 'React')
        - YEARS (경력 연수, 숫자)
        - SKILL_LEVEL (숙련도, '상', '중', '하')

        [조건]
        1. 반드시 Oracle SQL 쿼리만 한 줄로 출력해.
        2. 설명, 마크다운(```sql), 인사말은 절대 포함하지 마. 오직 SELECT로 시작하는 문장만 출력해.
        3. 문자열 비교는 대소문자를 구분하지 않게 UPPER()를 사용하거나 정확한 값을 사용해.
        4. 결과는 중복되지 않게 SELECT DISTINCT EMP_ID FROM EMP_SKILL... 형식을 사용해.

        사용자 질문: {question}
        """)

        # 5. LLM 실행
        chain = prompt | llm
        print("2. LLM 실행 중...")
        response = chain.invoke({"question": user_question})

        # 모델의 응답에서 불필요한 공백이나 마크다운 제거
        generated_sql = response.content.replace("```sql", "").replace("```", "").strip()
        print(f"3. 생성된 SQL: {generated_sql}")

        # 6. 스프링부트로 전송
        print(f"4. 스프링부트로 요청 전송: {SPRING_BOOT_URL}")
        try:
            # 헤더를 명시적으로 추가하여 JSON임을 알림
            headers = {'Content-Type': 'application/json'}
            spring_response = requests.post(
                SPRING_BOOT_URL,
                json={"sql": generated_sql},
                headers=headers,
                timeout=10
            )

            print(f"5. 스프링 응답 상태 코드: {spring_response.status_code}")
            print(f"6. 스프링 응답 본문(Raw): '{spring_response.text}'") # 여기가 비어있으면 문제!

            if not spring_response.text or spring_response.status_code != 200:
                return jsonify({
                    "status": "error",
                    "message": f"스프링 서버 응답 실패 (Code: {spring_response.status_code})",
                    "sql": generated_sql
                }), 500

            db_result = spring_response.json()
        except Exception as spring_err:
            print(f"스프링부트 통신 에러: {spring_err}")
            return jsonify({"status": "error", "message": "스프링 서버와 통신할 수 없습니다."}), 502

        return jsonify({
            "status": "success",
            "question": user_question,
            "generated_sql": generated_sql,
            "data": db_result
        })

    except Exception as e:
        print(f"최종 에러 발생: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
# 블루프린트 등록
app.register_blueprint(bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)