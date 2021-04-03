# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 19:02:56 2021

@author: Maryam Ghodsvali
"""

from set_path import path, sheets
import pandas as pd

class policy:
    def __init__(self):
        self.read_policy_dict()
    
    def read_policy_dict(self):
        df = pd.read_excel(path.path_to_meta, sheet_name = sheets.policy_dict, header=None).T
        self.policy_dict = {df.iloc[0,i]: int(df.iloc[1,i]) for i in range(len(df.columns))}