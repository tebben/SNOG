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

crt = model()
clc = crt.clc
cmb = clc.cmb
lu = clc.lu
```




**Citation**

We are currently working on a journal publication for the developed spatial nexus optimization model. Meanwhile, if you have used our framework for research purposes, please cite us with:
