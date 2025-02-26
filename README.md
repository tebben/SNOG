# Spatial Multi-objective Optimization of Food-Water-Energy Nexus in Python  

The Spatial Nexus Optimization model is proposed as a frame of reference aiming to address the assessment of fundamental requirements for a balanced nexus system combined with a number of particular policy actions on social and environmental implications of uncontrolled resource use. 
The presented tool can: 
- accommodate inputs that are particular to a given context;
- yield results in a geographically understandable layout;
- be uncomplicated from an analytical point of view, while supplying a comprehensive view of the situation; and
- test realistic options.

Through the model, decision-makers are provided with choices of adjustable technological, environmental, and social policies to develop and validate various possible scenarios for the nexus process. Policies can be assigned in combination or individually to a location of desire, and possible implications in socio-ecological systems performance can be discussed simultaneously. Thus, optimal choices of nexus policies considering future implications can be made, along with a spatially validated action plan.

The model is developed based on a modified version of Non-dominated Sortng Genetic Algorith-II (NSGA-II) using [Pymoo](https://pymoo.org/index.html).

**Calculating propeties for a given policy map**

The model can accomodate any land use, train the optimized policy map, and calculate relevant properties. But our aim here is to show how the default land use (i.e., case study of this research) can be loaded and the pproperties can be calculated.

First we import the model and load it with default parameters. The model contains various objects, so we assign them to different variables for convinience.

```
from model import model

snog = model() # main model module
clc = snog.clc # module to calculate the properties
cmb = clc.cmb # module that provides support for the calculation of optimization parameters for all possible combinations of policies
lu = clc.lu # module containing the land-use information
```

The landuse map is a 2-dimensional numpy array. We can read the current landuse as follows:

```
landuse_map = lu.landuse2d
```

We coded Null (not applicable) cells as -1.

We can read the landuse legend in a dictionary as follows:

```
landuse_legend = lu.landuse_dict
```

We can also read the landuse in a 1-dimensional array excluding -1 values.

```
landuse_flat = lu.landuse
```

You need to provide a policy map - an array with the same shape as the 2-dimensional landuse - to be able to calculate the properties.
Policy map should contain integer values with a certain range. You can derive the shape and the range of values for the policy map as below.

```
policy_shape = lu.landuse_shape
policy_range = cmb.policy_range
```

The policy map - k - with the above specification should be an input from the user. For illustration purpose, we generate k with random numbers.

```
k = lu.make_2d(clc.get_random_k())
```
For any not-applicable cell we use -1, and that means the cell is out of the lanuse boundary.

Values within policy_range are the policies for each cell in three categories:
- 1 to 10: Base policies.
- 11: Neutral policy
- 12 to max(policy_range): Combined policies.

Base policies are the actual policies. To derive names and the characteristics of the Base policies, we can call the following methods.

```
from policy import policy

pl = policy() # initializing the policy object
policy_legend = pl.policy_dict # read the name of the Base policies in a dictionary
policy_characteristics = pl.policy_characteristics # read the policy characteristics in a pandas dataframe. Index are the policies and columns are the characteristics.
```

Combined policies are combinations of two or more Base policies. For example, the user might choose policy 1 and 3 for a cell, but the cell should be coded as 12. Here are the possible policy combinations for the default scenario:

| Coded value | Actual policy combination |
| ------ | ------ |
| 12 | 1, 3 |
| 13 | 1, 4 |
| 14 | 2, 3 |
| 15 | 2, 3, 9 |
| 16 | 5, 6 |
| 17 | 5, 6, 7 |
| 18 | 7, 8 |
| 19 | 5, 6, 7, 8 |

Choosing any other combination considers invalid.

Policy 11 is called Neutral policy and that basically means no policy for the specific cell. Therefore, any cell without policy should be coded as 11.

It is also possible to load the pre-trained optimized policy map for the case study.

```
from optimized import optimized

k = optimized().read()
```

Until now, k is a 2-dimensional array, but in order to use it, we need to filter out the -1 values and make it 1-dimensional

```
k = k[lu.landuse_mask]
```

Now that we have the policy map ready, we can calculate the following properties:

```
climate_stress_control = clc.CLIMATE_STRESS_CONTROL(k)
nexus_resilience = clc.NEXUS_RESILIENCE(k)
social_ecological_integrity = clc.SOCIAL_ECOLOGICAL_INTEGRITY(k)
```

To calculate supply demand balance in percentage for food, water, and energy, a dictionary "sd" with the following properties is needed:
keys are the policy number from 1 to 10 (int) - values are number of cells that each policy is assigned to (int).
E.g., having one cell assigned to each policy:

```
sd = {k:1 for k in range(1,11)}
food, water, energy = clc.supply_demand_balance(sd)
```

**Invalid user policy choice**

Some policies are not compatible with some types of landuse. To see what are the compatible policies for each landuse, the following method can be called.

The result is a pandas.DataFrame. The index are the landuse types and the columns are the policies. 1 indicates that the policy is compatible with the landuse.

```
compatibility_metrix = cmb.compat_df
```

Some policies can be combined. To see what combinations of pilicies are possible, the following method can be called.

The result is a Dictionary. The keys are Combined policy codes, and values are Base policy codes involved in the combination.

```
combinations = cmb.combination
```


**Contact**

Feel free to contact me if you have any question:
Maryam Ghodsvali (m.ghodsevali [at] gmail.com)
