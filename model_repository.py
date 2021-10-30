from model import model
from configs import model_vars as v
from optimized import optimized
from policy import policy
import numpy as np

class ModelRepository():

    def __init__(self):
        self.models = {}
        self.addModel("bsd", [5.58233839, 51.47146286], model(), optimized().read(), {"1": 1, "2": 1, "3": 1, "4": 1, "5": 34, "6": 23, "7": 13, "8": 31, "9": 1, "10": 0 }, policy(), v)

    def addModel(self, id, topLeft, model, optimized, optimizedPlanning, policy, vars):
        policyGrid = np.array(optimized)
        k = policyGrid[model.clc.lu.landuse_mask]

        self.models[id] = {
            "gridTopLeft": topLeft,
            "model": model,
            "optimizedPlanning": optimizedPlanning,
            "optimizedPolicyGrid": optimized,
            "optimizedClimateStressControl": model.clc.CLIMATE_STRESS_CONTROL(k),
            "optimizedNexusResilience": model.clc.NEXUS_RESILIENCE(k),
            "optimizedSocialEcologicalIntegrity": model.clc.SOCIAL_ECOLOGICAL_INTEGRITY(k),
            "policy": policy,
            "vars": vars
        }

    def getAll(self):
        return self.models

    def getByID(self, id: str):
        if not id in self.models:
            return None

        return self.models[id]

    def getModelIDs(self):
        return list(self.models.keys())
