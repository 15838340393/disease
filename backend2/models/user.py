import uuid
from flask_sqlalchemy import SQLAlchemy

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

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
        }
