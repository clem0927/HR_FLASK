# src/__init__.py
from flask import Flask
from flask_cors import CORS


def create_app():
    app = Flask(__name__)

    # CORS 설정 (Spring Boot 서버에서 호출 가능하도록)
    CORS(app)

    # Young 모듈
    from src.young.chatbot.routes import bp as chatbot_bp
    from src.young.regression.routes import bp as regression_bp

    # Gyu 모듈 - 포상 추천 AI
    from src.gyu.routes import bp as gyu_reward_bp

    app.register_blueprint(chatbot_bp)
    app.register_blueprint(regression_bp)
    app.register_blueprint(gyu_reward_bp)

    return app
