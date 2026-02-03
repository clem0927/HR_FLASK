# src/__init__.py
from flask import Flask
from flask_cors import CORS


def create_app():
    app = Flask(__name__)
    CORS(app, supports_credentials=True, origins=["http://localhost:5173"])

    from src.young.chatbot.routes import bp as chatbot_bp
    from src.eun.attendance.routes import bp as attendance_bp
    from src.hyun.search.routes import bp as search_bp
    from src.young.phaseAi.routes import bp as phaseAi_bp

    # Gyu 모듈 - 포상 추천 AI
    from src.gyu.routes import bp as gyu_reward_bp

    app.register_blueprint(chatbot_bp)
    app.register_blueprint(gyu_reward_bp)
    app.register_blueprint(attendance_bp)
    app.register_blueprint(search_bp)
    app.register_blueprint(phaseAi_bp)

    return app
