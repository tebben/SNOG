# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 19:02:56 2021

@author: Maryam Ghodsvali
"""
from set_path import path, sheets
import pandas as pd
    
def extract_vars(df):
    ks = []
    vs = []
    for index, row in df.iterrows():
        k = row[0].strip()
        if type(row[1]) == str and '*' in row[1]:
            v = float(row[1].split('*')[0])*float(row[1].split('*')[1])
        else:
            v = float(row[1])
        ks.append(k)
        vs.append(v)
    return ks, vs

class model_vars:
    p = path.path_to_meta
    df = pd.read_excel(p, sheet_name = sheets.configs, header = None)
    ks, vs = extract_vars(df)
    for k,v in zip(ks, vs):
        if k == 'vegetable_demand':
            v = v*population
        exec("%s = %s" %(k, str(v)))
