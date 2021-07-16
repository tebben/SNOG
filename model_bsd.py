from model import model
from policy import policy
from optimized import optimized

class ModelBSD():

    def __init__(self):
        self.snog = model() # main model module
        self.clc = self.snog.clc # module to calculate the properties
        self.cmb = self.clc.cmb # module that provides support for the calculation of optimization parameters for all possible combinations of policies
        self.lu = self.clc.lu # module containing the land-use information

        # The landuse map is a 2-dimensional numpy array. We can read the current landuse as follows
        self.landuse_map = self.lu.landuse2d

        # We can read the landuse legend in a dictionary as follows
        self.landuse_legend = self.lu.landuse_dict

        # We can also read the landuse in a 1-dimensional array excluding -1 values
        self.landuse_flat = self.lu.landuse

        # You need to provide a policy map - an array with the same shape as the 2-dimensional landuse -
        # to be able to calculate the properties.
        # Policy map should contain integer values with a certain range.
        # You can derive the shape and the range of values for the policy map as below
        self.policy_shape = self.lu.landuse_shape
        self.policy_range = self.cmb.policy_range
        #print(policy_range) # integer

        # The policy map - k - with the above specification should be an input from the user.
        # For illustration purpose, we generate k with random numbers
        self.k = self.lu.make_2d(self.clc.get_random_k())

        # 1 to 10: Base policies.
        # 11: Neutral policy
        # 12 to max(policy_range): Combined policies.
        #print(self.cmb.combination)

        # Base policies are the actual policies.
        # To derive names and the characteristics of the Base policies, we can call the following methods
        self.pl = policy() # initializing the policy object
        self.policy_legend = self.pl.policy_dict # read the name of the Base policies in a dictionary
        self.policy_characteristics = self.pl.policy_characteristics # read the policy characteristics in a pandas dataframe. Index are the policies and columns are the characteristics.
        k = optimized().read()

        # Until now, k is a 2-dimensional array, but in order to use it,
        # we need to filter out the -1 values and make it 1-dimensional
        k = k[self.lu.landuse_mask]

        # Now that we have the policy map ready, we can calculate the following properties
        climate_stress_control = self.clc.CLIMATE_STRESS_CONTROL(k) # higher value, better climate stress control
        nexus_resilience = self.clc.NEXUS_RESILIENCE(k) # higher value, better nexus resilience
        social_ecological_integrity = self.clc.SOCIAL_ECOLOGICAL_INTEGRITY(k) # higher value, betteer social-ecological integrity
        print(climate_stress_control)

    def getProperties(self):
        # It is also possible to load the pre-trained optimized policy map for the case study
        k = optimized().read()

        # Until now, k is a 2-dimensional array, but in order to use it,
        # we need to filter out the -1 values and make it 1-dimensional
        k = k[self.lu.landuse_mask]

        # Now that we have the policy map ready, we can calculate the following properties
        climate_stress_control = self.clc.CLIMATE_STRESS_CONTROL(k) # higher value, better climate stress control
        nexus_resilience = self.clc.NEXUS_RESILIENCE(k) # higher value, better nexus resilience
        social_ecological_integrity = self.clc.SOCIAL_ECOLOGICAL_INTEGRITY(k) # higher value, betteer social-ecological integrity

