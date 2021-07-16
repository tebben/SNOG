import json

from model_repository import ModelRepository
from rest.exceptions import InvalidGrid, ModelNotFound

class ModelService:
    """Service for handling all requests to the model endpoint"""

    def __init__(self, modelRepository: ModelRepository):
        self.__repo = modelRepository
    
    def getModelList(self):
        return { "models":  self.__repo.getModelNames()}

    def getModelByID(self, id):
        model = self.__repo.getByID(id)
        if model is None:
            raise ModelNotFound()
        
        return { "model": id }

    def calculateProperties(self, model: str):
        """Calculate properties for a given model and grid policies
        Args:
            model (str): Name of the model
        Returns:
            result: JSON ...
        Raises:

        """

        return json.loads("{\"hello\": \"World\"}")

