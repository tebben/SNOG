import json

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
        policy = modelConfig["policy"]

        print(model.clc.lu.landuse_shape)

        return {
            "id": id,
            "gridTopLeft": modelConfig["gridTopLeft"],
            "gridSize": modelConfig["gridSize"],
            "landuseMap": model.clc.lu.landuse2d.tolist(),
            "landuseLegend": model.clc.lu.landuse_dict,
            "policyShape": model.clc.lu.landuse_shape,
            "policyRange": list(model.clc.cmb.policy_range),
            "policyLegend": policy.policy_dict,
            "policyCharacteristics": json.loads(policy.policy_characteristics.to_json())
        }

    def calculateProperties(self, id: str):
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
        optimized = modelConfig["optimized"]
        # Until now, k is a 2-dimensional array, but in order to use it,
        # we need to filter out the -1 values and make it 1-dimensional
        k = optimized[model.clc.lu.landuse_mask]

        # Now that we have the policy map ready, we can calculate the following properties
        climate_stress_control = model.clc.CLIMATE_STRESS_CONTROL(k) # higher value, better climate stress control
        nexus_resilience = model.clc.NEXUS_RESILIENCE(k) # higher value, better nexus resilience
        social_ecological_integrity = model.clc.SOCIAL_ECOLOGICAL_INTEGRITY(k) # higher value, betteer social-ecological integrity
        
        return {
            "climateStressControl": climate_stress_control,
            "nexusResilience": nexus_resilience,
            "socialEcologicalIntegrity": social_ecological_integrity
        }
