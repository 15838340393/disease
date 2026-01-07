from datetime import datetime
import uuid
from werkzeug.security import generate_password_hash, check_password_hash

from models import db


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    uuid = db.Column(
        db.String(36),
        unique=True,
        nullable=False,
        default=lambda: str(uuid.uuid4())
    )

    username = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)

    password_hash = db.Column(db.String(255), nullable=False)

    # 角色：user / admin
    role = db.Column(db.String(20), nullable=False, default="user")

    # 状态：active / disabled
    status = db.Column(db.String(20), nullable=False, default="active")

    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            "id": self.id,
            "uuid": self.uuid,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
