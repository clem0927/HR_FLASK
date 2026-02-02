# ==============================
# 한글 텍스트 정규화
# ==============================
def normalize_korean_text(text: str) -> str:
    if not text:
        return ""
    # 모든 공백을 제거하여 "일정 탭에서 오류" -> "일정탭에서오류"로 만듦
    return "".join(text.split())