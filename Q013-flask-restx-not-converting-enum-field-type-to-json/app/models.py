import enum
# from app import db
import sqlalchemy as db
from typing import List
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class eGameLevel(enum.Enum):
    BEGINNER = "Beginner"
    ADVANCED = "Advanced"


class Game(Base):
    __tablename__ = "game_stage"

    id = db.Column(db.Integer(), primary_key=True)
    game_level = db.Column(
        db.Enum(eGameLevel), default=eGameLevel.BEGINNER, nullable=False
    )
    user_id = db.Column(
        db.Integer(), db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    # user = relationship("User", backref="game__level_submissions", lazy=True)

    def __init__(self, game_level, user_id):
        self.game_level = game_level
        self.user_id = user_id

    def __repr__(self):
        return "Game(game_level%s, " "user_id%s" % (self.game_level, self.user_id)

    def json(self):
        return {"game_level": self.game_level, "user_id": self.user_id}

    @classmethod
    def by_game_id(cls, _id):
        return cls.query.filter_by(id=_id)

    @classmethod
    def find_by_game_level(cls, game_level):
        return cls.query.filter_by(game_level=game_level)

    @classmethod
    def by_user_id(cls, _user_id):
        return cls.query.filter_by(user_id=_user_id)

    @classmethod
    def find_all(cls) -> List["Game"]:
        return cls.query.all()
        # return db.session.query(cls).all()

    def save_to_db(self) -> None:
        db.session.add(self)
        db.session.commit()

    def delete_from_db(self) -> None:
        db.session.delete(self)
        db.session.commit()
