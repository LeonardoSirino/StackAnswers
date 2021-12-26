from enum import Enum

import sqlalchemy as sa
from flask import Flask
from flask_restx import Api, Namespace, Resource
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker


class eGameLevel(str, Enum):
    BEGINNER = "Beginner"
    ADVANCED = "Advanced"


engine = sa.create_engine("sqlite:///:memory:")
session = scoped_session(sessionmaker(bind=engine))
Base = declarative_base()


class Game(Base):
    __tablename__ = "game"
    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    level = sa.Column(sa.Enum(eGameLevel), default=eGameLevel.BEGINNER, nullable=False)

    def __repr__(self):
        return f"Game(id={self.id}, level={self.level})"

    def json(self):
        data = {"id": self.id, "level": self.level}
        return data


Base.metadata.create_all(engine)

g1 = Game(level=eGameLevel.BEGINNER)
g2 = Game(level=eGameLevel.ADVANCED)

session.add_all([g1, g2])
session.commit()
query_content = session.query(Game).all()

games_ns = Namespace("Game Requests", description="All Games Requests")


app = Flask(__name__)
api = Api(app)


@api.route("/game")
class GameRequestsListAPI(Resource):
    @games_ns.doc("Get all Game requests.")
    def get(self):
        data = [x.json() for x in query_content]
        return data, 200


app.run(debug=True)
