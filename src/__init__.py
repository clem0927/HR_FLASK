# src/__init__.py
from flask import Flask

def create_app():
    app = Flask(__name__)

    from src.young.chatbot.routes import bp as chatbot_bp
    from src.young.regression.routes import bp as regression_bp

    app.register_blueprint(chatbot_bp)
    app.register_blueprint(regression_bp)

    return app
