from flask import Blueprint, request, jsonify
from models import db, User, Prediction

# 如果你已经接入 JWT，可取消注释
# from flask_jwt_extended import jwt_required, get_jwt, get_jwt_identity

from ml.predict_service import predict_from_artifact


def _save_prediction(disease: str, version: str, features: dict, result: dict):
    """
    写入预测记录。即使失败也不影响主流程（预测依然返回）。
    user_uuid 暂时从 features 取（你后面接 JWT 后再改为 token 里取）。
    """
    try:
        user_uuid = features.get("user_uuid")
        user = None
        if user_uuid:
            user = User.query.filter_by(uuid=user_uuid).first()

        p = Prediction(
            user_id=user.id if user else None,
            user_uuid=user.uuid if user else user_uuid,
            disease=disease,
            version=version,
            probability=result["probability"],
            threshold=result["threshold"],
            pred_label=result["pred_label"],
            risk_level=result["risk_level"],
            features=features,
        )
        db.session.add(p)
        db.session.commit()
        return p
    except Exception:
        db.session.rollback()
        return None


predict_bp = Blueprint("predict", __name__)

HEART_BASIC_FILE = "heart_basic_pipeline.joblib"
HEART_ADV_FILE = "heart_pipeline.joblib"


@predict_bp.route("/heart/basic", methods=["POST"])
# @jwt_required()  # 需要登录再开
def predict_heart_basic():
    data = request.get_json() or {}
    features = data.get("features") or data  # 支持两种格式：{features:{...}} 或直接 {...}

    result, err = predict_from_artifact(HEART_BASIC_FILE, features)
    if err:
        return jsonify(err), 400
    user_uuid = features.get("user_uuid")  # 可选：前端传，或以后 JWT 里取
    user = None
    if user_uuid:
        user = User.query.filter_by(uuid=user_uuid).first()

    p = Prediction(
        user_id=user.id if user else None,
        user_uuid=user.uuid if user else user_uuid,
        disease="heart",
        version="basic",
        probability=result["probability"],
        threshold=result["threshold"],
        pred_label=result["pred_label"],
        risk_level=result["risk_level"],
        features=features
    )
    db.session.add(p)
    db.session.commit()

    _save_prediction("heart", "basic", features, result)

    return jsonify({
        "disease": "heart",
        "version": "basic",
        "result": result
    })


@predict_bp.route("/heart/advanced", methods=["POST"])
# @jwt_required()  # 需要登录再开
def predict_heart_advanced():
    data = request.get_json() or {}
    features = data.get("features") or data

    result, err = predict_from_artifact(HEART_ADV_FILE, features)
    if err:
        return jsonify(err), 400
    user_uuid = features.get("user_uuid")
    user = None
    if user_uuid:
        user = User.query.filter_by(uuid=user_uuid).first()

    p = Prediction(
        user_id=user.id if user else None,
        user_uuid=user.uuid if user else user_uuid,
        disease="heart",
        version="advanced",
        probability=result["probability"],
        threshold=result["threshold"],
        pred_label=result["pred_label"],
        risk_level=result["risk_level"],
        features=features
    )
    db.session.add(p)
    db.session.commit()

    _save_prediction("heart", "advanced", features, result)

    return jsonify({
        "disease": "heart",
        "version": "advanced",
        "result": result
    })


@predict_bp.route("/history", methods=["GET"])
def list_predictions():
    user_uuid = request.args.get("user_uuid")
    q = Prediction.query.order_by(Prediction.created_at.desc())
    if user_uuid:
        q = q.filter_by(user_uuid=user_uuid)

    items = q.limit(100).all()
    return jsonify([p.to_dict() for p in items])


@predict_bp.route("/history", methods=["GET"])
def predict_history():
    """
    GET /api/predict/history?user_uuid=xxx&disease=heart&version=basic&page=1&page_size=20
    """
    user_uuid = request.args.get("user_uuid")
    disease = request.args.get("disease")
    version = request.args.get("version")

    page = int(request.args.get("page", 1))
    page_size = int(request.args.get("page_size", 20))
    page_size = min(max(page_size, 1), 100)

    q = Prediction.query

    if user_uuid:
        q = q.filter(Prediction.user_uuid == user_uuid)
    if disease:
        q = q.filter(Prediction.disease == disease)
    if version:
        q = q.filter(Prediction.version == version)

    q = q.order_by(Prediction.created_at.desc())

    pagination = q.paginate(page=page, per_page=page_size, error_out=False)
    items = [p.to_dict() for p in pagination.items]

    return jsonify({
        "page": page,
        "page_size": page_size,
        "total": pagination.total,
        "items": items
    })
