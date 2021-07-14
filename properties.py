# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 19:02:56 2021

@author: Maryam Ghodsvali
"""

# First we import the model and load it with default parameters.
# The model contains various objects, so we assign them to different variables for convinience.

from model import model
from policy import policy
from optimized import optimized

snog = model() # main model module
clc = snog.clc # module to calculate the properties
cmb = clc.cmb # module that provides support for the calculation of optimization parameters for all possible combinations of policies
lu = clc.lu # module containing the land-use information

# The landuse map is a 2-dimensional numpy array. We can read the current landuse as follows
landuse_map = lu.landuse2d

# We can read the landuse legend in a dictionary as follows
landuse_legend = lu.landuse_dict

# We can also read the landuse in a 1-dimensional array excluding -1 values
landuse_flat = lu.landuse

# You need to provide a policy map - an array with the same shape as the 2-dimensional landuse -
# to be able to calculate the properties.
# Policy map should contain integer values with a certain range.
# You can derive the shape and the range of values for the policy map as below
policy_shape = lu.landuse_shape
policy_range = cmb.policy_range
print(policy_range) # integer

# The policy map - k - with the above specification should be an input from the user.
# For illustration purpose, we generate k with random numbers
k = lu.make_2d(clc.get_random_k())

# 1 to 10: Base policies.
# 11: Neutral policy
# 12 to max(policy_range): Combined policies.
print(cmb.combination)

# Base policies are the actual policies.
# To derive names and the characteristics of the Base policies, we can call the following methods
pl = policy() # initializing the policy object
policy_legend = pl.policy_dict # read the name of the Base policies in a dictionary
policy_characteristics = pl.policy_characteristics # read the policy characteristics in a pandas dataframe. Index are the policies and columns are the characteristics.

# It is also possible to load the pre-trained optimized policy map for the case study
k = optimized().read()

# Until now, k is a 2-dimensional array, but in order to use it,
# we need to filter out the -1 values and make it 1-dimensional
k = k[lu.landuse_mask]

# Now that we have the policy map ready, we can calculate the following properties
climate_stress_control = clc.CLIMATE_STRESS_CONTROL(k) # higher value, better climate stress control
nexus_resilience = clc.NEXUS_RESILIENCE(k) # higher value, better nexus resilience
social_ecological_integrity = clc.SOCIAL_ECOLOGICAL_INTEGRITY(k) # higher value, betteer social-ecological integrity

# Invalid user input
# Compatibility - DataFrame
compatibility_metrix = cmb.compat_df
# Combinations - Dictionary
combinations = cmb.combination



