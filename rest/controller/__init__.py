from rest import app
from model_repository import ModelRepository

from rest.service.model import ModelService

modelRepository = ModelRepository()
modelService = ModelService(modelRepository)
