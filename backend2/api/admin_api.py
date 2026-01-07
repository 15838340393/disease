from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt

from models import db, User

admin_bp = Blueprint("admin", __name__)


def _require_admin():
    claims = get_jwt()
    if claims.get("role") != "admin":
        return False
    return True


@admin_bp.route("/users", methods=["GET"])
@jwt_required()
def list_users_admin():
    if not _require_admin():
        return jsonify({"msg": "admin required"}), 403

    users = User.query.order_by(User.id.desc()).all()
    return jsonify([u.to_dict() for u in users])


@admin_bp.route("/users/<string:user_uuid>", methods=["PATCH"])
@jwt_required()
def update_user_admin(user_uuid):
    if not _require_admin():
        return jsonify({"msg": "admin required"}), 403

    user = User.query.filter_by(uuid=user_uuid).first()
    if not user:
        return jsonify({"msg": "user not found"}), 404

    data = request.get_json() or {}

    # 允许改 role/status
    if "role" in data:
        if data["role"] not in ("user", "admin"):
            return jsonify({"msg": "role must be user or admin"}), 400
        user.role = data["role"]

    if "status" in data:
        if data["status"] not in ("active", "disabled"):
            return jsonify({"msg": "status must be active or disabled"}), 400
        user.status = data["status"]

    db.session.commit()
    return jsonify(user.to_dict())
