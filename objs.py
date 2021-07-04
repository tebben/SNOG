# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 19:02:56 2021

@author: Maryam Ghodsvali
"""
from configs import model_vars as v

def beta():
    beta_dict={
        1: lambda x: 59.42 * v.grid_lenght**2,
        2: lambda x: 141658.60 * v.grid_lenght**2,
        3: lambda x: 0.00001 * v.grid_lenght,
        4: lambda x: 0,
        5: lambda x: 2.18 * v.grid_lenght**2,
        6: lambda x: 0.00016 * 365 * v.grid_lenght**2 if x == 0 else (
            0.0024 * 365 * v.grid_lenght**2 if x == 2 else (
            0.0006 * 365 * v.grid_lenght**2 if x == 1 else 1)),
        # 7: 0.038 * grid_lenght**2,
        7: lambda x: 332.09 * v.grid_lenght**2, 
        8: lambda x: 0,
        # 9: 2.19 * grid_lenght**2,
        9: lambda x: 19179.14 * v.grid_lenght**2,
        # 10: 0.02 * grid_lenght**2,
        10: lambda x: 174.15 * v.grid_lenght**2,
        11: lambda x: 0
        }
    return beta_dict

def alpha():
    alpha_dict={
        1: 0.086,
        2: 0.035,
        3: 0.093,
        4: 0.080,
        5: 0.119,
        6: 0.052,
        7: 0.264,
        8: 0.278,
        9: 0.048,
        10: 0.026,
        11: 0
        }
    return alpha_dict

def climate_stress_control():
    climate_dict = {
        1: 75 * v.grid_lenght**2,
        2: 14.4 * v.grid_lenght**2,
        3: 0.08 * v.grid_lenght**2,
        4: 0.0122 * v.grid_lenght**2,
        5: 5.09 * v.grid_lenght**2,
        6: 0.0122 * v.grid_lenght**2,
        7: 17.52 * v.grid_lenght**2,
        8: 21.87 * v.grid_lenght**2,
        9: 15.7 * v.grid_lenght**2,
        10: 14.89 * v.grid_lenght**2,
        11: 0
        }
    return climate_dict