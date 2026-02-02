# src/young/chatbot/ollama_client.py

import requests

model_name = "gemma3:4b"
api_url = "http://localhost:11434/api/chat"

def ask_ollama(user_message: str, context_text: str, chat_history: list[dict]) -> str:
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
