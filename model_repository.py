from model import model
from optimized import optimized
from policy import policy

class ModelRepository():

    def __init__(self):
        self.models = {}

        self.addModel("bsd", [168608.1590000018, 386901.9517999999], 100, model(), optimized().read(), policy())

    def addModel(self, id, topLeft, gridSize, model, optimized, policy):
        self.models[id] = {
            "gridTopLeft": topLeft,
            "gridSize": gridSize,
            "model": model,
            "optimized": optimized,
            "policy": policy
        }

    def getAll(self):
        return self.models

    def getByID(self, id: str):
        if not id in self.models:
            return None

        return self.models[id]

    def getModelIDs(self):
        return list(self.models.keys())
