from flask import Blueprint, request, jsonify
from models import db, User

user_bp = Blueprint("user", __name__)


# api/user_api.py（create_user 里替换为下面逻辑）
@user_bp.route("/", methods=["POST"])
def create_user():
    data = request.get_json() or {}

    username = data.get("username")
    password = data.get("password")
    email = data.get("email")

    if not username or not password:
        return jsonify({"msg": "username and password are required"}), 400

    # 防重复（username/email）
    if User.query.filter_by(username=username).first():
        return jsonify({"msg": "username already exists"}), 400
    if email and User.query.filter_by(email=email).first():
        return jsonify({"msg": "email already exists"}), 400

    user = User(username=username, email=email, role="user", status="active")
    user.set_password(password)

    db.session.add(user)
    db.session.commit()

    return jsonify(user.to_dict()), 201



@user_bp.route("/", methods=["GET"])
def list_users():
    users = User.query.all()
    return jsonify([u.to_dict() for u in users])


@user_bp.route("/<string:user_uuid>", methods=["GET"])
def get_user(user_uuid):
    user = User.query.filter_by(uuid=user_uuid).first()
    if not user:
        return jsonify({"msg": "user not found"}), 404
    return jsonify(user.to_dict())


@user_bp.route("/<string:user_uuid>", methods=["PUT"])
def update_user(user_uuid):
    user = User.query.filter_by(uuid=user_uuid).first()
    if not user:
        return jsonify({"msg": "user not found"}), 404

    data = request.get_json()
    user.username = data.get("username", user.username)
    db.session.commit()

    return jsonify(user.to_dict())


@user_bp.route("/<string:user_uuid>", methods=["DELETE"])
def delete_user(user_uuid):
    user = User.query.filter_by(uuid=user_uuid).first()
    if not user:
        return jsonify({"msg": "user not found"}), 404

    db.session.delete(user)
    db.session.commit()

    return jsonify({"msg": "deleted"})
