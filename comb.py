# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 19:02:56 2021

@author: Maryam Ghodsvali
"""

from set_path import path, sheets
import pandas as pd
import numpy as np
from compat import compatibility as compat
from dist import distance as dist
from objs import alpha, beta

class combination:
    def __init__(self):
        self.read_combination()
        self.compat_combination()
        self.distans_combination()
        self.alpha_combination()
        self.beta_combination()
    
    def read_combination(self):
        df = pd.read_excel(path.path_to_meta, sheet_name = sheets.combination, header=None).T
        cor = lambda x: int(x.strip())
        self.combination = {df.iloc[0,i]: [*map(cor, df.iloc[1,i].split(','))] for i in range(len(df.columns))}

    def compat_combination(self):
        compatibility = compat().compatibility
        compatibility.columns = range(1, len(compatibility.columns)+1)
        for key,value in self.combination.items():
            z = compatibility[value[0]].copy()
            for i in value[1:]: 
                z *= compatibility[i]
            compatibility[key] = z
        swap = np.vectorize(lambda x : int(not bool(x)))
        compatibility = swap(compatibility)
        self.compat = compatibility
    
    def alpha_combination(self):
        self.alpha_dict = alpha()
        for key,value in self.combination.items():
            self.alpha_dict[key]=np.prod([*map(lambda x: self.alpha_dict[x], value)])

    def beta_combination(self):
        self.beta_dict = beta()
        for key, value in self.beta_dict.items():
            self.beta_dict[key] = [value]
        for key,value in self.combination.items():
            self.beta_dict[key]=[*map(lambda x: self.beta_dict[x][0], value)]
        
    def distans_combination(self):
        distance = dist().distance
        for key, value in self.combination.items():
            r = distance[value[0]]
            for val in value[1:]:
                tmp = distance[val]
                if len(tmp.shape) == 1:
                    distance[(key,'')] = r.combine(distance[val], max)
                else:
                    for col in range(distance[val].shape[1]):
                        col_name = distance[val].columns[col]
                        distance[(key,col_name)] = r.combine(distance[val].iloc[:,col], max)
        for col in [x for x in distance.columns if x[0] in self.combination.keys()]:
            distance.loc[col,:] = ''
        for index, row in distance.iterrows():
            for col in distance.columns:
                if not row[col] == '':
                    distance.loc[col, index] = row[col]
        for index in [x for x in distance.columns if x[0] in self.combination.keys()]:
            for col in [x for x in distance.columns if x[0] in self.combination.keys()]:
                combs = []
                set1 = self.combination[index[0]]
                set2 = self.combination[col[0]]
                for s1 in set1:
                    for s2 in set2:
                        tmp = distance.loc[[s1], [s2]]
                        if tmp.shape == (1,1):
                            combs.append(tmp.iloc[0,0])
                        else:
                            pref_index = index[1] if index[1] in tmp.index.get_level_values(1) else ''
                            pref_col = col[1] if col[1] in tmp.columns.get_level_values(1) else ''
                            combs.append(tmp.loc[(s1, pref_index),(s2, pref_col)])
                res = max(combs)
                distance.loc[index, col] = res
        self.dist = distance       