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
        self.read_policy_comb()
    
    def read_policy_dict(self):
        df = pd.read_excel(path.path_to_meta, sheet_name = sheets.policy_dict, header=None).T
        self.policy_dict = {df.iloc[0,i]: int(df.iloc[1,i]) for i in range(len(df.columns))}
        
    def read_policy_comb(self):
        df = pd.read_excel(path.path_to_meta, sheet_name = sheets.policy_comb, skiprows=1).iloc[:,2:]
        self.scenarios = [(s.strip(), {index+1:x+1 for index,x in enumerate(df[df[s] == 1].index.tolist())}) for s in df.columns]
        
