# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 19:02:56 2021

@author: Maryam Ghodsvali
"""

from model import model

pop_size = 100
norm = True
filename_suffix = ''

nsga2 = model(pop_size, norm, filename_suffix)
nsga2.run()

# Results
nsga2.get_objective_space()
nsga2.get_policy_map()
nsga2.get_G_plot()
nsga2.save_results()
