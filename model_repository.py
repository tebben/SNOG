from model import model
from optimized import optimized
from policy import policy
import numpy as np

class ModelRepository():

    def __init__(self):
        self.models = {}
        self.addModel("bsd", [5.58233839, 51.47146286], 100, model(), optimized().read(), policy())

    def addModel(self, id, topLeft, gridSize, model, optimized, policy):
        policyGrid = np.array(optimized)
        k = policyGrid[model.clc.lu.landuse_mask]

        self.models[id] = {
            "gridTopLeft": topLeft,
            "gridSize": gridSize,
            "model": model,
            "optimizedPolicyGrid": optimized,
            "optimizedClimateStressControl": model.clc.CLIMATE_STRESS_CONTROL(k),
            "optimizedNexusResilience": model.clc.NEXUS_RESILIENCE(k),
            "optimizedSocialEcologicalIntegrity": model.clc.SOCIAL_ECOLOGICAL_INTEGRITY(k),
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
