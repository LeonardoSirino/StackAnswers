from flask import Flask
from flask_restx import Api, Namespace, Resource, fields

from app.models import Game
from app.schemas import GameSchema

app = Flask(__name__)
api = Api(app)


GAME_REQUEST_NOT_FOUND = "Game request not found."
GAME_REQUEST_ALREADY_EXSISTS = "Game request '{}' Already exists."

game_ns = Namespace("Game", description="Available Game Requests")
games_ns = Namespace("Game Requests", description="All Games Requests")


game_schema = GameSchema()
games_list_schema = GameSchema(many=True)


gamerequest = game_ns.model(
    "Game",
    {
        "game_level": fields.String("Game Level: Must be one of: BEGINNER, ADVANCED."),
        "user_id": fields.Integer,
    },
)


@api.route("/game")
class GameRequestsListAPI(Resource):
    @games_ns.doc("Get all Game requests.")
    def get(self):
        return games_list_schema.dump(Game.find_all()), 200

    @games_ns.expect(gamerequest)
    @games_ns.doc("Create a Game request.")
    def post(self):
        print(request)
        game_json = request.get_json()
        game_data = game_schema.load(game_json)

        print(game_data)
        game_data.save_to_db()

        return game_schema.dump(game_data), 201


if __name__ == "__main__":
    app.run(debug=True)
