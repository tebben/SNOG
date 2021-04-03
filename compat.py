# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 19:02:56 2021

@author: Maryam Ghodsvali
"""

from set_path import path, sheets
import pandas as pd

class compatibility:
    def __init__(self):
        self.read_compatibility()
    
    def read_compatibility(self):
        self.compatibility = pd.read_excel(path.path_to_meta, sheet_name=sheets.compatibility, index_col=0)
