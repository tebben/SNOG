from model_bsd import ModelBSD

class ModelRepository():

    def __init__(self):
        self.models = {
            "bsd": ModelBSD()
        }

    def getAll(self):
        return self.models

    def getByID(self, id: str):
        if not id in self.models:
            return None

        return self.models

    def getModelNames(self):
        return list(self.models.keys())
