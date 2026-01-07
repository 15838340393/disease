import os
from dotenv import load_dotenv
from flask_jwt_extended import JWTManager

base_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(base_dir, ".env"))

from flask import Flask
from flask_migrate import Migrate
from config import Config
from models import db
from api.user_api import user_bp
from api.auth_api import auth_bp
from api.admin_api import admin_bp
from api.predict_api import predict_bp

def create_app():
    app = Flask(__name__)
    # 加载配置
    app.config.from_object(Config)
    print("JWT_SECRET_KEY =", app.config.get("JWT_SECRET_KEY"))

    # 初始化数据库
    db.init_app(app)
    Migrate(app, db)
    # JWT
    JWTManager(app)
    # 注册蓝图（路由）
    app.register_blueprint(user_bp, url_prefix="/api/users")
    app.register_blueprint(auth_bp, url_prefix="/api/auth")
    app.register_blueprint(admin_bp, url_prefix="/api/admin")
    app.register_blueprint(predict_bp, url_prefix="/api/predict")
    return app


app = create_app()

if __name__ == "__main__":
    print(app.url_map)
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
