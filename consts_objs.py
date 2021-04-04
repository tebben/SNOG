# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 19:02:56 2021

@author: Maryam Ghodsvali
"""

from set_path import path
import pandas as pd
import numpy as np
from landuse import landuse_creator
from configs import model_vars as v
from comb import combination as combination_obj
from itertools import product

class consts_objs:
    def __init__(self, scenario, norm: bool = False):
        self.iter = 500 # iterations to normalize values
        self.cmb = combination_obj(scenario)
        self.lu = landuse_creator()
        self.landuse2d = self.lu.read2d_partial_random()
        self.landuse = self.lu.read()
        self.compatibility_const_vec() # create func needed for G1
        self.beta_vec()
        self.max_alpha = max(self.cmb.alpha_dict.values())
        self.alpha_vec()
        self.normalize(norm) # read normalized values or perform normalization
        self.translate()
    
    def normalize(self, norm):
        if not norm:
            df = pd.read_csv(path.path_to_norm_vals, header = None).T
            self.norm = {df.iloc[0,i]: int(df.iloc[1,i]) for i in range(len(df.columns))}
        else:
            self.norm = dotdict({'g1':1,
                                 'g2':1,
                                 'g3':1,
                                 'g4':1,
                                 'g5':1,
                                 'g6':1,
                                 'f1':1,
                                 'f2':1})
            gs = self.normalizer_consts()
            fs = self.normalizer_obj()
            self.norm = {'g1':abs(gs[0]),
                         'g2':abs(gs[1]),
                         'g3':abs(gs[2]),
                         'g4':abs(gs[3]),
                         'g5':abs(gs[4]),
                         'g6':abs(gs[5]),
                         'f1':abs(fs[0]),
                         'f2':abs(fs[1])}
            df = pd.DataFrame(self.norm, index= [0]).T
            df.to_csv(path.path_to_norm_vals, header = None)
        self.norm = dotdict(self.norm)
            
    def get_random_k(self):
        k = np.random.randint(low = 1, high = max(self.cmb.trans.keys())+1, size = self.lu.landuse.shape)
        k = self.translator(k)
        return k
    
    def translate(self):
        self.translate = np.vectorize(lambda x: self.cmb.trans[x])
    
    def translator(self, k):
        return self.translate(k)
    
    def normalizer_consts(self):
        res = []
        for i in range(self.iter):
            print('\rNormalizing Constraints: %d/%d'%(i+1,self.iter),end="")
            k = self.get_random_k()
            res.append([self.G1(k), self.G2(k), self.G3(k), self.G4(k), self.G5(k), self.G6(k)])
        print()
        means = []
        for i in range(len(res[0])):
            means.append(round(np.mean([a[i] for a in res])))
        return means
        
    def normalizer_obj(self):
        res = []
        for i in range(self.iter):
            print('\rNormalizing Objectives: %d/%d'%(i+1,self.iter),end="")
            k = self.get_random_k()
            res.append([self.F1(k), self.F2(k)])
        print()
        means = []
        for i in range(len(res[0])):
            means.append(round(np.mean([a[i] for a in res])))
        return means
    
    def calc_beta(self, k):
        return self.beta_func(k, self.landuse).sum()
    
    def beta_vec(self):
        self.beta_func = np.vectorize(self.beta_base) 
    
    def beta_base(self, k, landuse):
        return sum([func(landuse) for func in self.cmb.beta_dict[k]])
    
    def calc_alpha(self, k):
        return self.alpha_func(k).sum()
    
    def alpha_vec(self):
        self.alpha_func = np.vectorize(self.alpha_base)
    
    def alpha_base(self, k):
        return self.max_alpha - self.cmb.alpha_dict[k]
    
    def compatibility_const(self, k, landuse):
        out = self.cmb.compat[landuse, int(k-1)]
        return out
    
    def compatibility_const_vec(self):
        self.compatibility_func = np.vectorize(self.compatibility_const)

    def calc_g1(self, k):
        return self.compatibility_func(k, self.landuse).sum()#/k.shape[0]
    
    def spatial_adjacency(self, k):
        k = self.lu.make_2d(k)
        res_dict = {}
        for row in self.cmb.dist.index:
            for column in self.cmb.dist.columns:
                min_dist = self.cmb.dist.loc[row, column]
                if not min_dist == 0:
                    matched_1 = self.select_index(row, k)
                    matched_2 = self.select_index(column, k)
                    res_dict[(row, column)] = self.pairwise_distance(matched_1, matched_2, min_dist)
                else:
                    res_dict[(row, column)] = 0
        res = sum(res_dict.values())
        return res 
    
    def pairwise_distance(self, matched_1, matched_2, min_dist):
        prod = [*product(matched_1, matched_2)]
        dists = np.array([*map(self.calc_dist, prod)])
        dists = dists < min_dist
        return dists.sum()
    
    def select_index(self, ind, k):
        if ind[1] == '': # no landuse:
            match = np.where(k == ind[0])
            match = [*zip(match[0], match[1])]
        else:
            match1 = np.where(k == ind[0])
            match2 = np.where(self.lu.landuse2d ==  ind[1])
            match = list(set([*zip(match1[0], match1[1])])&set([*zip(match2[0], match2[1])]))
        return match
    
    def calc_dist(self, points):
        p1 = points[0]
        p2 = points[1]
        return np.sqrt((p1[1]-p2[1])**2+(p1[0]-p2[0])**2)
    
    def G1(self, k):
        g1 = self.calc_g1(k)/self.norm.g1 - v.error_buffer
        return g1
    
    def G2(self, k):
        g2 = -((np.count_nonzero(k == 1) + np.count_nonzero(k == 12) + np.count_nonzero(k == 13)) \
                * v.grid_lenght**2 * 25 - v.vegetable_demand)/self.norm.g2 - v.error_buffer
        return g2
    
    def G3(self, k):
        g3 = ((np.count_nonzero(k == 1) + np.count_nonzero(k == 2) + np.count_nonzero(k == 12) \
                + np.count_nonzero(k == 13) + np.count_nonzero(k == 14) + np.count_nonzero(k == 15)) \
              * v.grid_lenght**2 - v.agri_land_availability)/self.norm.g3 - v.error_buffer
        return g3
    
    def G4(self, k):
        g4 = self.spatial_adjacency(k)/self.norm.g4 - v.error_buffer
        return g4
    
    def G5(self, k):
        g5 = ((np.count_nonzero(k == 7) + np.count_nonzero(k == 9) + np.count_nonzero(k == 10) \
                + np.count_nonzero(k == 15) + np.count_nonzero(k == 17) + np.count_nonzero(k == 18) \
                    + np.count_nonzero(k == 19)) * v.grid_lenght**2 - v.energy_land_availabilty)/self.norm.g5 - v.error_buffer
        return g5
    
    def G6(self, k):
        g6 = -(((((np.count_nonzero(k == 7) + np.count_nonzero(k == 17) + np.count_nonzero(k == 18) + np.count_nonzero(k == 19)) * 0.017) \
                + ((np.count_nonzero(k == 9) + np.count_nonzero(k == 15)) * 0.41) + (np.count_nonzero(k == 10) * 0.014)) \
                    * v.grid_lenght**2 * 24* 365) - v.electricity_demand)/self.norm.g6 - v.error_buffer
        return g6
    
    def F1(self, k):
        return self.calc_beta(k)/self.norm.f1
        
    def F2(self, k):
        return self.calc_alpha(k)/self.norm.f2
        
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__    