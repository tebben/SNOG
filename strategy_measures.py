# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 14:59:43 2021

@author: 20180742
"""

# climate stress control
def co2reduction():
    co2reduction_dict={
        1: 75,
        2: 14.4,
        3: 0.08,
        4: 0.0122,
        5: 5.09,
        6: 0.0122,
        7: 17.52,
        8: 21.87,
        9: 15.7,
        10: 14.89,
        11: 0
        }
    return co2reduction_dict
# combinations = sum

def M1(co2reduction_dict, k):
    m1 = sum(co2reduction(k) * v.grid_lenght**2)
    return m1

def nexusresilience():
    x = f1,
    y = f2
    return f1,f2

def socialecologicalintegrity():
    sci = 0 - gs
    return sci