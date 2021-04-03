# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 19:02:56 2021

@author: Maryam Ghodsvali
"""

from set_path import path, sheets
import pandas as pd
from landuse import landuse_creator
from configs import model_vars as v

class distance:
    def __init__(self):
        self.lu = landuse_creator()
        self.read_distancet()
    
    def read_distancet(self):
        distance = pd.read_excel(path.path_to_meta, sheet_name= sheets.distance, index_col=None, header = None)
        distance.fillna('',inplace = True)
        distance = distance.replace(self.lu.landuse_dict)
        distance.iloc[0,:] = distance.iloc[0,:].apply(lambda x: int(x.split('K')[-1]) if not x =='' else '')
        distance.iloc[:,0] = distance.iloc[:,0].apply(lambda x: int(x.split('K')[-1]) if not x =='' else '')
        distance = distance.set_index([0,1])
        columns = pd.MultiIndex.from_arrays(distance.iloc[0:2].values)
        distance = distance.iloc[2:]
        distance.columns = columns
        for index,row in distance.iterrows():
            for col in distance.columns:
                if not row[col] == '':
                    distance.loc[col,index] = row[col]
        self.distance = distance.applymap(lambda x: x/v.grid_lenght)