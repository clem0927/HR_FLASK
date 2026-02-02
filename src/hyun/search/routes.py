from flask import Flask, Blueprint, request, jsonify
from flask_cors import CORS
import requests
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)
# 리액트(3000번 포트 등)와의 통신을 위한 CORS 설정
CORS(app, resources={r"/*": {"origins": "*"}})

# 1. Blueprint 설정
bp = Blueprint("hyun_search", __name__, url_prefix="/hyun/search")

# 2. Ollama 모델 설정 (Gemma 3 4B)
llm = ChatOllama(
    model="gemma3:4b",
    temperature=0, # 일관된 결과를 위해 0으로 설정
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

        # 4. 프롬프트 구성 (해설 가이드 강화)
        prompt = ChatPromptTemplate.from_template("""
        너는 Oracle SQL 전문가야. 반드시 아래의 [직급 매핑 표]와 내용들을 준수하여 사용자의 질문을 SQL로 변환해.
        
        [직급 매핑 표]
        - '사장', '대표' -> EMP_ROLE = 'CEO'
        - '인사', '인사담당관' -> EMP_ROLE = 'HR'
        - '근태', '근태담당관' -> EMP_ROLE = 'ATTENDANCE'
        - '일정', '일정담당관' -> EMP_ROLE = 'SCHEDULE'
        - '평가' '평가담당관' -> EMP_ROLE = 'EVAL'
        - '보상' '보상담당관' -> EMP_ROLE = 'REWARD'
        - '팀장', '리더', '부장' -> EMP_ROLE = 'LEADER'
        - '시니어', '과장', '대리' -> EMP_ROLE = 'SENIOR'
        - '주니어', '신입', '사원급' -> EMP_ROLE = 'JUNIOR'
        
        [컬럼 구분 규칙 - 중요!]
        1. 기술 종류(Java, Python, Oracle 등) -> SKILL_NAME 컬럼 사용
        2. 숙련도(상, 중, 하) -> SKILL_LEVEL 컬럼 사용 (예: s.SKILL_LEVEL = '상')
        3. 경력 연수(숫자) -> YEARS 컬럼 사용
        
        [용어 변환 규칙]
        - SKILL_NAME은 반드시 영문 대문자로 변환 (예: '자바' -> 'JAVA', '파이썬' -> 'PYTHON')
        - '사람', '직원'이라는 단어는 특정 직급이 아닌 '전체 직원'을 의미하므로 직급(EMP_ROLE) 조건을 걸지 마라.
        단 "팀장급 직원", "근태담당 사람" 등 구체적인 직급 명칭이 사용된 경우 해당 조건을 포함해라.
        
        [테이블 정보]
        - DEPT (부서) : DEPT_NO, DEPT_NAME, DEPT_LOC
        - EMP (사원): EMP_ID, EMP_NAME, EMP_ROLE, DEPT_NO 
        - EMP_SKILL (기술): EMP_ID, SKILL_NAME, SKILL_LEVEL, YEARS
        
        [작성 규칙]
        0. 모든 SQL은 반드시 Oracle 21c 문법을 따른다.
        1. 기본적으로 SELECT e.* FROM EMP e 를 사용해 사원테이블을 조회하는 쿼리문을 작성한다.
        2. JOIN 대신 EXISTS 또는 IN 구문을 사용한다.
        3. 위의 [직급 매핑 표]와 [용어 변환 규칙]을 따라서 입력값으로 사용한다.
        4. 기술/숙련도/연차 조건은 반드시 기술 테이블(EMP_SKILL)을 사용하여 사원테이블을 조회할 것
        5. "검색 조건(SKILL_NAME 등)에는 반드시 UPPER()를 사용하여 대소문자 구분 없이 검색되게 하라. (예: UPPER(s.SKILL_NAME) = 'JAVA')"
        6. 복잡한 집계가 불가능하면 SQL에 'ERROR'라고 적어라.
        
        [응답 형식 - 중요!]
        SQL: <한 줄의 SQL>
        EXPLANATION: <실행한 SQL에 대한 매우 간단한 한 줄의 한국어 해설, 단 테이블에 대한 해설은 제외함>
        
        사용자 질문: {question}
        """)

        # 5. LLM 실행 (파이프 연산자 활용)
        chain = prompt | llm
        print("2. LLM 실행 중...")
        response = chain.invoke({"question": user_question})
        content = response.content.strip()

        # 6. SQL 및 해설 파싱 (텍스트 분리 로직)
        generated_sql = ""
        explanation = ""

        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.upper().startswith("SQL:"):
                generated_sql = line[4:].strip().replace("```sql", "").replace("```", "")
            elif line.upper().startswith("EXPLANATION:"):
                explanation = line[12:].strip()

        # 모델이 'ERROR'를 반환했거나 파싱에 실패한 경우
        if "ERROR" in generated_sql.upper() or not generated_sql:
            return jsonify({
                "status": "fail",
                "question": user_question,
                "explanation": "죄송합니다. 질문이 너무 복잡하여 분석이 어렵거나 해당되지 않는 질문입니다.",
                "data": []
            })

        print(f"3. 생성된 SQL: {generated_sql}")
        print(f"3-1. 생성된 해설: {explanation}")

        # 7. 스프링부트로 요청 전송
        print(f"4. 스프링부트로 요청 전송")
        try:
            headers = {'Content-Type': 'application/json'}
            spring_response = requests.post(
                SPRING_BOOT_URL,
                json={"sql": generated_sql},
                headers=headers,
                timeout=10
            )

            # 스프링 서버에서 SQL 실행 오류가 난 경우 (모델이 잘못된 쿼리를 생성했을 때)
            if spring_response.status_code != 200:
                return jsonify({
                    "status": "fail",
                    "question": user_question,
                    "explanation": "질문을 SQL로 변환하는 데 성공했으나, 실행 중 오류가 발생했습니다. 조금 더 명확하게 질문해 주세요.",
                    "data": []
                })

            db_result = spring_response.json()
        except Exception as spring_err:
            print(f"스프링부트 통신 에러: {spring_err}")
            return jsonify({"status": "error", "message": "스프링 서버와 통신할 수 없습니다."}), 502

        # 8. 최종 결과 반환
        return jsonify({
            "status": "success",
            "question": user_question,
            "explanation": explanation,
            "data": db_result
        })

    except Exception as e:
        print(f"최종 에러 발생: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# 블루프린트 등록
app.register_blueprint(bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)