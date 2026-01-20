# 통합서버 실행
from src import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
