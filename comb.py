# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 19:02:56 2021

@author: Maryam Ghodsvali
"""

from set_path import path, sheets
import pandas as pd
import numpy as np
from compat import compatibility as compat
from landuse import landuse_creator
from dist import distance as dist
from objs import alpha, beta, climate_stress_control
from policy import policy

class combination:
    def __init__(self, scenario):
        self.lu = landuse_creator()
        self.scenario = scenario
        self.policy_comb()
        self.select_scenario()
        self.read_combination()
        self.compat_combination()
        self.compat_dataframe()
        self.distance_combination()
        self.adj4scenario_combination() # Update scenario and combination with combined policies
        # self.adj4scenario_compatibility()
        # self.adj4scenario_distance()
        self.alpha_combination()
        self.beta_combination()
        self.climate_combination()
        
    def select_scenario(self):
        try:
            if type(self.scenario) == int:
                self.scenario = self.scenarios[self.scenario][0]
            self.scenario = [x for x in self.scenarios if x[0] == self.scenario][0]
        except:
            raise ValueError('Incorrect scenario "%s"' %self.scenario)
    
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
        compatibility = self.manual_changes_compatibility(compatibility)
        swap = np.vectorize(lambda x : int(not bool(x)))
        compatibility = swap(compatibility)
        self.compat = compatibility
        
    def compat_dataframe(self):
        cols = list(range(1, self.compat.shape[1]+1))
        rows = list(self.lu.landuse_dict.keys())
        rows = [x for x in rows if not x =='N.A.']
        df = pd.DataFrame(self.compat, columns = cols, index=rows)
        df = df.replace(1,2).replace(0,1).replace(2,0)
        self.compat_df = df
    
    def manual_changes_compatibility(self, compatibility):
        compatibility.loc[:, 3] = 0 # K3 only valid in combination with K1 or K2
        compatibility.loc[['M','C','G'], 4] = 0 # K4 for M, C and G is valid only in combination with K1
        compatibility.loc[:, 9] = 0 # K9 only valid in combination with K1 or K2
        return compatibility
    
    def adj4scenario_compatibility(self):
        self.compat = self.compat[:,self.valids]
    
    def adj4scenario_combination(self):
        self.combination = {k:v for k,v in self.combination.items() if all(x in self.scenario[1].values() for x in v)}
        m = max(self.scenario[1].keys())
        comb_dict = list(self.combination.items())
        comb_dict = [list(x) for x in comb_dict]
        help_key = [x[0] for x in comb_dict]
        comb_dict = [x for _,x in sorted(zip(help_key,comb_dict))]
        trans = {}
        for i in range(len(comb_dict)):
            trans[m+1] = comb_dict[i][0]
            comb_dict[i][0] = m+1
            m+=1
        self.scenario[1].update(trans)
        self.valids = np.array(list(self.scenario[1].values()))-1
        self.trans = self.scenario[1]
        self.scenario = self.scenario[0]
    
    def alpha_combination(self):
        self.alpha_dict = alpha()
        for key,value in self.combination.items():
            self.alpha_dict[key]=sum([*map(lambda x: self.alpha_dict[x], value)])

    def beta_combination(self):
        self.beta_dict = beta()
        for key, value in self.beta_dict.items():
            self.beta_dict[key] = [value]
        for key,value in self.combination.items():
            self.beta_dict[key]=[*map(lambda x: self.beta_dict[x][0], value)]
        
    def climate_combination(self):
        self.climate_dict = climate_stress_control()
        for key,value in self.combination.items():
            self.climate_dict[key]=sum([*map(lambda x: self.climate_dict[x], value)])
    
    def distance_combination(self):
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
        
    def adj4scenario_distance(self):
        self.dist = self.dist.iloc[self.dist.index.get_level_values(0).isin(self.valids+1), self.dist.columns.get_level_values(0).isin(self.valids+1)]
    
    def policy_comb(self):
        self.scenarios = policy().scenarios
        
    @property
    def policy_range(self):
        return range(1, max(self.trans.keys())+1)
