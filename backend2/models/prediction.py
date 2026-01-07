import uuid as uuid_lib
from datetime import datetime
from models import db


class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    # 关联用户（允许为空：未登录也能预测）
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    user_uuid = db.Column(db.String(36), nullable=True, index=True)

    disease = db.Column(db.String(32), nullable=False)      # heart / diabetes / stroke
    version = db.Column(db.String(32), nullable=False)      # basic / advanced

    probability = db.Column(db.Float, nullable=False)
    threshold = db.Column(db.Float, nullable=False)
    pred_label = db.Column(db.Integer, nullable=False)      # 0/1
    risk_level = db.Column(db.String(16), nullable=False)   # low/medium/high

    # 保存一次预测的输入（便于复盘与可视化）
    features = db.Column(db.JSON, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "user_uuid": self.user_uuid,
            "disease": self.disease,
            "version": self.version,
            "probability": self.probability,
            "threshold": self.threshold,
            "pred_label": self.pred_label,
            "risk_level": self.risk_level,
            "features": self.features,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
