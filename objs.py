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

def supply_demand_balance(sd):
    valid = range(1,11)
    if not all(x in sd.keys() for x in valid):
        raise ValueError(f'Key values should contain {list(valid)}')
    sdd = {
        1: (25*0.07*100*sd[1]*10000)/(v.vegetable_demand*v.population),
        2: (0.02316*12*100*sd[2]*10000)/(0.101*v.population*365),
        3: (25*0.07*100*sd[3]*10000)/(v.vegetable_demand*v.population),
        4: (1*0.09*0.763*100*sd[4]*10000)/(v.water_demand),
        5: (0.817*100*sd[5]*10000)/(v.water_demand),
        6: (((0.115 + 0.442 + 1.69)/3)*0.0013*0.9*100*365*sd[6]*10000)/(v.water_demand),
        7: (300*0.92*100*sd[7]*10000)/(v.electricity_demand),
        8: (150*100*sd[8]*10000)/(v.electricity_demand),
        9: (0.09*92*8*100*sd[9]*10000)/(v.electricity_demand),
        10: (0.014*365*24*100*sd[10]*10000)/(v.electricity_demand),
        }
    food = sdd[1] + sdd[2] + sdd[3]
    water = sdd[4] + sdd[5] + sdd[6]
    energy = sdd[7] + sdd[8] + sdd[9] + sdd[10]
    food = 100 if food > 100 else round(food)
    water = 100 if water > 100 else round(water)
    energy = 100 if energy > 100 else round(energy)
    return food, water, energy