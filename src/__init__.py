# src/__init__.py
from flask import Flask
from flask_cors import CORS


def create_app():
    app = Flask(__name__)
    CORS(app, supports_credentials=True, origins=["http://localhost:5173"])

    from src.young.chatbot.routes import bp as chatbot_bp
    from src.young.regression.routes import bp as regression_bp
    from src.eun.attendance.routes import bp as attendance_bp

    app.register_blueprint(chatbot_bp)
    app.register_blueprint(regression_bp)
    app.register_blueprint(attendance_bp)

    return app
