from flask_restx import fields
from rest import app

model_list = app.api.model('ModelList', {
    "models": fields.List(fields.String, required=True, desctiption="List of available models"),
})

properties_get = app.api.model('PropertiesGet', {
    "grid": fields.List(fields.List(fields.Integer), required=True, desctiption="List of available models", ),
})