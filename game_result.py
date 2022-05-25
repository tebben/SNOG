# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 10:37:29 2022

@author: 20180742
"""

from model import model
import pandas as pd
import numpy as np
import ast

def extract_policies(map_, lu):
    array = np.array(ast.literal_eval(map_))
    array = array[lu.landuse_mask]
    return array
    
def count_policies(array, landuse, combinations):
    comb = pd.DataFrame({'lu':landuse, 'policy':array})
    to_add = []
    to_remove = []
    for index, row in comb.iterrows():
        policy = row['policy']
        if policy in combinations:
            for sub_policy in combinations[policy]:
                tmp = row.copy()
                tmp['policy'] = sub_policy
                to_add.append(tmp)
            to_remove.append(index)
    comb_new = comb.copy()
    comb_new = comb_new.drop(to_remove)
    comb_new = comb_new.append(pd.DataFrame(to_add))
    comb_new = comb_new.reset_index(drop=True)
    comb_new = comb_new.groupby(['lu', 'policy']).size()
    comb_new.columns = ['count']
    return comb_new


def run():          
            
    path = r"C:\Users\20180742\OneDrive - TU Eindhoven\Collaboration\Papers\05-User Interface\Game Test\Results\Game Database\22.03.23_SNOG Scenarios Database.xlsx"
    
    snog = model()
    clc = snog.clc
    cmb = clc.cmb
    lu = clc.lu
    
    landuse = lu.landuse
    combinations = cmb.combination
    landuse_legend = lu.landuse_dict
    df = pd.read_excel(path, sheet_name = 'Data for analysis')
    
    col = 'What were your criteria for the BSD plan design?'
    df[col] = df[col].apply(lambda x: x.split(';'))
    dict_ = {}
    for index, row in df.iterrows():
        map_ = row['map']
        array = extract_policies(map_, lu)
        comb_new = count_policies(array, landuse, combinations)
        criteria = row[col]
        for criterion in criteria:
            if criterion in dict_:
                dict_[criterion].append(comb_new)
            else:
                dict_[criterion] = [comb_new]
    final = {}
    for key, value in dict_.items():
        base = pd.DataFrame(value[0])
        for series in range(1, len(value)):
            base = pd.merge(base, pd.DataFrame(value[series]), left_index=True, right_index=True, how='outer')
        base.columns = range(base.shape[1])
        base = base.fillna(0)
        base[key] = base.mean(axis=1)
        final[key] = base[[key]]
    
    dfs = list(final.values())
    f = dfs[0]
    for d in range(1, len(dfs)):
        f = pd.merge(f, dfs[d], left_index=True, right_index=True, how='outer')
    f = f.fillna(0)
    landuse_legend = {v:k for k,v in landuse_legend.items()}
    path = r"C:\Users\20180742\OneDrive - TU Eindhoven\Collaboration\Papers\05-User Interface\Game Test\Results\Result Analysis\Result.xlsx"
    f = f.reset_index()
    f.lu = f.lu.replace(landuse_legend)
    f = f.set_index(['lu', 'policy'])
    f.to_excel(path)
