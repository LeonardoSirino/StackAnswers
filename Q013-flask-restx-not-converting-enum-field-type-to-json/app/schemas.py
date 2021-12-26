import marshmallow_sqlalchemy as ma
from app.models import Game


class GameSchema(ma.SQLAlchemyAutoSchema):
    game = ma.fields.Nested("GameSchema", many=True)

    class Meta:
        model = Game
        load_instance = True
        include_fk = True
