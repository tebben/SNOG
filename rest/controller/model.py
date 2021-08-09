import logging

from flask_restx import Resource, reqparse, fields

from rest import app, models
from rest.controller import modelService
from rest.exceptions import InvalidGrid, ModelNotFound

logger = logging.getLogger(__name__)

@app.ns_model.route('/')
class ModelListController(Resource):
    """Get a list of available models from the service and return JSON"""

    @app.api.doc(description="Get a list of available models", responses={200: "Ok"})
    @app.ns_model.doc(description='Get a list of available models')
    @app.api.marshal_with(models.model_list)
    def get(self):
        """Returns a list of available models"""

        return modelService.getModelList(), 200

@app.ns_model.route('/<string:model_id>/')
class ModelController(Resource):
    """Get model info"""

    @app.api.doc(description="Get model info", responses={
        200: "Ok",
        404: "Model not found"})
    @app.ns_model.doc(description='Get model info')
    def get(self, model_id):
        """Get model info"""
        
        try:
            model = modelService.getModelByID(model_id)
        except ModelNotFound as err:
            return err.message, err.statusCode

        return model, 200

parser = reqparse.RequestParser()
parser.add_argument('grid', location='json', type=list)

@app.ns_model.route('/<string:model_name>/properties/')
class PropertiesController(Resource):
    """Get properties for a given model"""

    @app.api.doc(description="Get properties for a model", responses={
        200: "Ok",
        404: "Model not found",
        400: "Invalid grid"})
    @app.api.expect(models.properties_get, validate=True)
    def post(self, model_name):
        """Get properties for a model"""

        try:
            args = parser.parse_args()
            grid = args['grid']
            result = modelService.calculateProperties(model_name, grid)
        except ModelNotFound as err:
            return "Model '{}' not found".format(model_name), err.statusCode

        return result, 200