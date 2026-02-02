# src/__init__.py
from flask import Flask

def create_app():
    app = Flask(__name__)

    from src.young.chatbot.routes import bp as chatbot_bp
    from src.hyun.search.routes import bp as search_bp
    from src.young.phaseAi.routes import bp as phaseAi_bp

    app.register_blueprint(chatbot_bp)
    app.register_blueprint(search_bp)
    app.register_blueprint(phaseAi_bp)

    return app
