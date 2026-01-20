from flask import Blueprint

bp = Blueprint(
    "young_chatbot",
    __name__,
    url_prefix="/young/chatbot"
)

@bp.route("/ask")
def ask():
    return "chatbot ask"
