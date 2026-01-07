from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token

from models import User

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json() or {}

    login_type = data.get("loginType")  # "user" or "admin"
    account = data.get("account")       # username 或 email
    password = data.get("password")

    if login_type not in ("user", "admin"):
        return jsonify({"msg": "loginType must be user or admin"}), 400
    if not account or not password:
        return jsonify({"msg": "account and password are required"}), 400

    # 允许 username/email 登录
    user = User.query.filter((User.username == account) | (User.email == account)).first()
    if not user:
        return jsonify({"msg": "invalid account or password"}), 401

    if user.status != "active":
        return jsonify({"msg": "user is disabled"}), 403

    if not user.check_password(password):
        return jsonify({"msg": "invalid account or password"}), 401

    # 关键：管理员登录入口必须是 admin 角色
    if login_type == "admin" and user.role != "admin":
        return jsonify({"msg": "admin account required"}), 403

    additional_claims = {"role": user.role, "uuid": user.uuid}
    access_token = create_access_token(identity=str(user.id), additional_claims=additional_claims)

    return jsonify({
        "access_token": access_token,
        "user": user.to_dict()
    })
