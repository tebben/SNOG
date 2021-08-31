import json
import numpy as np

from optimized import optimized
from model_repository import ModelRepository
from rest.exceptions import InvalidGrid, ModelNotFound

class ModelService:
    """Service for handling all requests to the model endpoint"""

    def __init__(self, modelRepository: ModelRepository):
        self.__repo = modelRepository
    
    def getModelList(self):
        return { "models":  self.__repo.getModelIDs()}

    def getModelByID(self, id):
        modelConfig = self.__repo.getByID(id)

        if modelConfig is None:
            raise ModelNotFound()

        model = modelConfig["model"]
        optimized = modelConfig["optimizedPolicyGrid"]
        policy = modelConfig["policy"]

        return {
            "id": id,
            "grid": {
                "topLeft": modelConfig["gridTopLeft"],
                "size": modelConfig["gridSize"],
            },
            "optimized": {
                "policyGrid": optimized.tolist(),
                "climateStressControl": modelConfig["optimizedClimateStressControl"],
                "nexusResilience": modelConfig["optimizedNexusResilience"],
                "ecologicalIntegrity": modelConfig["optimizedSocialEcologicalIntegrity"],
            },
            "landuse": {
                "map": model.clc.lu.landuse2d.tolist(),
                "legend": model.clc.lu.landuse_dict
            },
            "policy": {
                "shape": model.clc.lu.landuse_shape,
                "range": list(model.clc.cmb.policy_range),
                "combinations": model.clc.cmb.combination,
                "compatibility": json.loads(model.clc.cmb.compat_df.to_json()),
                "legend": policy.policy_dict,
                "characteristics": json.loads(policy.policy_characteristics.to_json())
            }
        }

    def calculateProperties(self, id: str, grid):
        """Calculate properties for a given model and grid policies
        Args:
            model (str): Name of the model
        Returns:
            result: JSON ...
        Raises:

        """

        modelConfig = self.__repo.getByID(id)
        if modelConfig is None:
            raise ModelNotFound()

        model = modelConfig["model"]
        #optimized = modelConfig["optimized"]
        #k = optimized[model.clc.lu.landuse_mask]

        # 2d list to numpy, flatten and remove -1 values
        policyGrid = np.array(grid)
        k = policyGrid[model.clc.lu.landuse_mask]

        # calculate
        climate_stress_control = model.clc.CLIMATE_STRESS_CONTROL(k) # higher value, better climate stress control
        nexus_resilience = model.clc.NEXUS_RESILIENCE(k) # higher value, better nexus resilience
        social_ecological_integrity = model.clc.SOCIAL_ECOLOGICAL_INTEGRITY(k) # higher value, betteer social-ecological integrity
        
        return {
            "climateStressControl": climate_stress_control,
            "nexusResilience": nexus_resilience,
            "socialEcologicalIntegrity": social_ecological_integrity
        }
