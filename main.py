# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 19:02:56 2021

@author: Maryam Ghodsvali
"""

from model import model

scenario = 1
pop_size = 10
norm = False
filename_suffix = ''

# Initialize the Model
nsga2 = model(scenario, pop_size, norm, filename_suffix)

# Run
nsga2.run()

# Results
nsga2.get_objective_space()
nsga2.get_policy_map()
nsga2.get_G_plot()
nsga2.save_results()
