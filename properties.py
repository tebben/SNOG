# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 19:02:56 2021

@author: Maryam Ghodsvali
"""

from model import model
from optimized import optimized
from policy import policy

# Initialize the Model
crt = model()
clc = crt.clc
cmb = clc.cmb
lu = clc.lu
pl = policy()

# Provide the policy map k: numpy.array of shape lu.landuse_shape
# Array should contains integers. Range of values should be cmb.policy_range
# Here we provide an array with random numbers
k = lu.make_2d(clc.get_random_k())

# We can also get the optimized policy map
k = optimized().read()

# The mask should be used to filter out the irrelevant cells
k = k[lu.landuse_mask]

# Now the following proprties can be calculated.
f1 = clc.F1(k)
f2 = clc.F2(k)
g1 = clc.G1(k)
g2 = clc.G2(k)
g3 = clc.G3(k)
g4 = clc.G4(k)
g5 = clc.G5(k)
g6 = clc.G6(k)
climate_stress_control = clc.CLIMATE_STRESS_CONTROL(k)
nexus_resilience = clc.NEXUS_RESILIENCE(k)
social_ecological_integrity = clc.SOCIAL_ECOLOGICAL_INTEGRITY(k)
