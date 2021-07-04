# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 19:02:56 2021

@author: Maryam Ghodsvali
"""

from set_path import path, sheets
import pandas as pd

class optimized:
    def __init__(self):
        pass
        
    def read(self):
        self.optimized = pd.read_excel(path.path_to_optimized, sheet_name=sheets.res_x_best, header=None).values
        return self.optimized