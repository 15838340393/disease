import os
from dotenv import load_dotenv

base_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(base_dir, ".env"))

from flask import Flask
from flask_migrate import Migrate
from config import Config
from models import db
from api.user_api import user_bp


def create_app():
    app = Flask(__name__)

    # 加载配置
    app.config.from_object(Config)

    print("DATABASE_URL =", os.getenv("DATABASE_URL"))
    print("SQLALCHEMY_DATABASE_URI =", app.config.get("SQLALCHEMY_DATABASE_URI"))

    # 初始化数据库
    db.init_app(app)

    Migrate(app, db)
    # 注册蓝图（路由）
    app.register_blueprint(user_bp, url_prefix="/api/users")

    return app


app = create_app()

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
