# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 19:02:56 2021

@author: Maryam Ghodsvali
"""

from model import model

scenario = 4
pop_size = 700
norm = True
filename_suffix = '_Edited_WeightedCompat'

# Initialize the Model
nsga2 = model(scenario, pop_size, norm, filename_suffix)

# Run
nsga2.run()

# Results
nsga2.get_objective_space()
nsga2.get_policy_map()
nsga2.get_G_plot()
nsga2.get_hypervolume()
nsga2.save_results()
