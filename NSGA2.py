# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 11:20:46 2021

@author: Maryam Ghodsvali

This script file is generated for Spatial multi-objective FWE nexus optimization:
Non-dominated Sortng Genetic Algorith-II (NSGA-II) using Pymoo.
"""
import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import rasterio
from matplotlib.font_manager import FontProperties
from itertools import product
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.problem import Problem
from pymoo.util.termination.default import MultiObjectiveDefaultTermination
from pymoo.optimize import minimize
# from pymoo.visualization.scatter import Scatter
from pymoo.factory import  get_sampling, get_crossover, get_mutation#, get_decomposition
from pymoo.performance_indicator.hv import Hypervolume
from pymoo.performance_indicator.igd import IGD

from landuse import landuse_creator


directory = r"C:\Users\20180742\OneDrive - TU Eindhoven\Collaboration\Papers\04-Decision Support System\MOO algorithm scripts\Run Test"

# def read_landuse():
#     p = r"C:\Users\20180742\OneDrive - TU Eindhoven\Collaboration\Papers\04-Decision Support System\Data\GIS\landuse.tif"
#     raster = rasterio.open(p)
#     raster = raster.read(1)
#     val_convert = np.vectorize(lambda x: -1 if x in [0,15] else x)
#     raster = val_convert(raster)
#     raster_flat = raster.reshape(raster.shape[0]*raster.shape[1])
#     mask = raster_flat==1
#     other = np.random.choice(3, np.sum(mask), p=[0.57, 0.37, 0.06])
#     raster_flat[mask] = other
#     return raster

# landuse2d = read_landuse()
# landuse_size = (landuse2d.shape[0], landuse2d.shape[1])
# landuse_mask = landuse2d!=-1
# landuse = landuse2d.reshape(landuse2d.shape[0]*landuse2d.shape[1])
# landuse = landuse[landuse!=-1]
# rev = np.vectorize(lambda x:not x)
# landuse_mask_rev = rev(landuse_mask)

lu = landuse_creator()
landuse2d = lu.read2d_partial_random()
landuse = lu.read()

# Variables:
population = 4200
vegetable_demand = 0.125 * population * 365
agri_land_availability = 27000
energy_land_availabilty = 140000
electricity_demand = 43200000
error_buffer = 0
grid_lenght = 40
# k = np.random.randint(low = 1, high = 19, size = landuse_size)
# original_gridsize = k.shape
# k = k.reshape(landuse2d.shape[0]*landuse2d.shape[1])

policy_dict = {
    'Urban gardening': 1,
    'Limited land allocation for fodder crops': 2,
    'Sustainable farming production system': 3,
    'Draining garden design': 4,
    'Rainwater harvesting': 5,
    'On-site wastewater purification': 6,
    'Solar power roofs': 7,
    'Energy-saving households behavior': 8,
    'Biomass efficiency improvement': 9,
    'Wind power': 10
    }

landuse_dict = {
    'N.A.': -1,
    'Residential': 0,
    'Industrial': 1,
    'Commercial': 2,
    'Green': 3
    }
# landuse = np.array([[3,3,1,3],
#                     [2,0,0,0],
#                     [1,0,0,0],
#                     [3,2,0,1]]).reshape(16)
# landuse = np.random.choice(4, 700)
def create_random_landuse():
    landuse = np.random.choice(4, 700, p=[0.16, 0.11, 0.02, 0.71])
    landuse2d = landuse.reshape(20,35)
    return landuse, landuse2d


  
combination = {
    12: [1,3],
    13: [1,4],
    14: [2,3],
    15: [2,3,9],
    16: [5,6],
    17: [5,6,7],
    18: [7,8],
    19: [5,6,7,8]
    }

def compat_combination(compatibility):
    compatibility.columns = range(1, len(compatibility.columns)+1)
    for key,value in combination.items():
        z = compatibility[value[0]].copy()
        for i in value[1:]: 
            z *= compatibility[i]
        compatibility[key] = z
    return compatibility.values
   
path = r"C:\Users\20180742\OneDrive - TU Eindhoven\Collaboration\Papers\04-Decision Support System\Data\Optimization Development Calculation\21.03.05_Optimization Development Calculations_M.Ghodsvali.xlsx" 
compatibility = pd.read_excel(path, sheet_name='Compatibility', index_col=0)
compatibility = compat_combination(compatibility)
swap = np.vectorize(lambda x : int(not bool(x)))
compatibility = swap(compatibility)

distance = pd.read_excel(path, sheet_name= 'Distance', index_col=None, header = None)

def distance_fill(distance):
    distance.fillna('',inplace = True)
    distance = distance.replace(landuse_dict)
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
    distance = distance.applymap(lambda x: x/grid_lenght)
    for key, value in combination.items():
        r = distance[value[0]]
        for val in value[1:]:
            tmp = distance[val]
            if len(tmp.shape) == 1:
                distance[(key,'')] = r.combine(distance[val], max)
            else:
                for col in range(distance[val].shape[1]):
                    col_name = distance[val].columns[col]
                    distance[(key,col_name)] = r.combine(distance[val].iloc[:,col], max)
    for col in [x for x in distance.columns if x[0] in combination.keys()]:
        distance.loc[col,:] = ''
    for index, row in distance.iterrows():
        for col in distance.columns:
            if not row[col] == '':
                distance.loc[col, index] = row[col]
    for index in [x for x in distance.columns if x[0] in combination.keys()]:
        for col in [x for x in distance.columns if x[0] in combination.keys()]:
            combs = []
            set1 = combination[index[0]]
            set2 = combination[col[0]]
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
    return distance

distance = distance_fill(distance)

def select_index(ind, k):
    if ind[1] == '': # no landuse:
        match = np.where(k == ind[0])
        match = [*zip(match[0], match[1])]
    else:
        match1 = np.where(k == ind[0])
        match2 = np.where(landuse2d ==  ind[1])
        match = list(set([*zip(match1[0], match1[1])])&set([*zip(match2[0], match2[1])]))
    return match

def pairwise_distance(matched_1, matched_2, min_dist):
    dists = []
    for x,y in matched_1:
        for i,j in matched_2:
            dist = np.sqrt((y-j)**2+(x-i)**2)
            if dist >= min_dist:
                dists.append(0)
            else:
                dists.append(1)
    return sum(dists)

def calc_dist(points):
    p1 = points[0]
    p2 = points[1]
    return np.sqrt((p1[1]-p2[1])**2+(p1[0]-p2[0])**2)

def pairwise_distance_2(matched_1, matched_2, min_dist):
    prod = [*product(matched_1, matched_2)]
    dists = np.array([*map(calc_dist, prod)])
    dists = dists < min_dist
    return dists.sum()
                
def spatial_adjacency(k):
    k = lu.make_2d(k)
    res_dict = {}
    for row in distance.index:
        for column in distance.columns:
            min_dist = distance.loc[row, column]
            if not min_dist == 0:
                matched_1 = select_index(row, k)
                matched_2 = select_index(column, k)
                res_dict[(row, column)] = pairwise_distance_2(matched_1, matched_2, min_dist)
            else:
                res_dict[(row, column)] = 0
    res = sum(res_dict.values())
    return res    

def beta(k, landuse):
    policy_6 = lambda a: 0.00016 * 365 * grid_lenght**2 if a == 0 else (
            0.0024 * 365 * grid_lenght**2 if a == 2 else (
            0.0006 * 365 * grid_lenght**2 if a == 1 else 1))
    beta_dict={
        1: 59.42 * grid_lenght**2,
        2: 141658.60 * grid_lenght**2,
        3: 0.00001 * grid_lenght,
        4: 0,
        5: 2.18 * grid_lenght**2,
        6: policy_6(landuse),
        # 7: 0.038 * grid_lenght**2,
        7: 332.09 * grid_lenght**2, 
        8: 0,
        # 9: 2.19 * grid_lenght**2,
        9: 19179.14 * grid_lenght**2,
        # 10: 0.02 * grid_lenght**2,
        10: 174.15 * grid_lenght**2,
        11: 0
        }
    beta_dict = beta_combination(beta_dict)
    out = beta_dict[k]
    return out

def beta_combination(beta_dict):
    for key,value in combination.items():
        beta_dict[key]=sum([*map(lambda x: beta_dict[x], value)])
    return beta_dict

def beta_vec(k):
    vec = np.vectorize(beta)
    out = vec(k,landuse).sum()
    return out

def alpha(k):
    alpha_dict={
        1: 0.086,
        2: 0.035,
        3: 0.093,
        4: 0.080,
        5: 0.119,
        6: 0.052,
        7: 0.264,
        8: 0.278,
        9: 0.048,
        10: 0.026,
        11: 0
        }
    alpha_dict = alpha_combination(alpha_dict)
    max_alpha = max(alpha_dict.values())
    out = max_alpha - alpha_dict[k]
    return out

def alpha_combination(alpha_dict):
    for key,value in combination.items():
        alpha_dict[key]=np.prod([*map(lambda x: alpha_dict[x], value)])
    return alpha_dict
        
def alpha_vec(k):
    vec = np.vectorize(alpha)
    out = vec(k).sum()
    return out

def compatibility_const(k, landuse):
    out = compatibility[landuse, int(k-1)]
    return out

def compatibility_const_vec(k):
    vec = np.vectorize(compatibility_const)
    out = vec(k, landuse).sum()
    return out

# def consts_normalizer():
#     res = []
#     for i in range(500):
#         print(i)
#         x = np.random.randint(low = 1, high = 19, size = landuse.shape)#.reshape(700)
#         # landuse2d = np.random.randint(low = 0, high = 4, size = (20,35))
#         # landuse = landuse2d.reshape(700)
#         g1 = (compatibility_const_vec(x)/x.shape[0])
#         g2 = -((np.count_nonzero(x == 1) + np.count_nonzero(x == 12) + np.count_nonzero(x == 13)) \
#                 * grid_lenght**2 * 25 - vegetable_demand)
#         g3 = ((np.count_nonzero(x == 1) + np.count_nonzero(x == 2) + np.count_nonzero(x == 12) \
#                 + np.count_nonzero(x == 13) + np.count_nonzero(x == 14) + np.count_nonzero(x == 15)) \
#               * grid_lenght**2 - agri_land_availability)
#         g4 = spatial_adjacency(x)
#         g5 = ((np.count_nonzero(x == 7) + np.count_nonzero(x == 9) + np.count_nonzero(x == 10) \
#                 + np.count_nonzero(x == 15) + np.count_nonzero(x == 17) + np.count_nonzero(x == 18) \
#                     + np.count_nonzero(x == 19)) * grid_lenght**2 - energy_land_availabilty)
#         g6 = -(((((np.count_nonzero(x == 7) + np.count_nonzero(x == 17) + np.count_nonzero(x == 18) + np.count_nonzero(x == 19)) * 0.017) \
#                 + ((np.count_nonzero(x == 9) + np.count_nonzero(x == 15)) * 0.41) + (np.count_nonzero(x == 10) * 0.014)) \
#                     * grid_lenght**2 * 24* 365) - electricity_demand)
#         res.append([g1,g2,g3,g4,g5,g6])
#     # g1s = [a[3] for a in res]
#     # plt.hist(g1s)
#     meds = []
#     for i in range(6):
#         meds.append(np.mean([a[i] for a in res]))

# def objective_normalizer():
#     res = []
#     for i in range(500):
#         print(i)
#         x = np.random.randint(low = 1, high = 19, size = landuse.shape)#.reshape(700)
#         # landuse2d = np.random.randint(low = 0, high = 4, size = (20,35))
#         # landuse = landuse2d.reshape(700)
#         f1 = beta_vec(x)
#         f2 = alpha_vec(x)
#         res.append([f1,f2])
#     meds = []
#     for i in range(2):
#         meds.append(np.mean([a[i] for a in res]))


class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=landuse.shape[0],
                         n_obj=2,
                         n_constr=6,
                         xl=np.array([1]*landuse.shape[0]),
                         xu=np.array([1]*landuse.shape[0])*max(combination.keys()),
                         type_var=int,
                         elementwise_evaluation=True)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = beta_vec(x)/33225194484
        f2 = alpha_vec(x)/217

        g1 = (compatibility_const_vec(x)/x.shape[0])/0.46 - error_buffer
        g2 = -((np.count_nonzero(x == 1) + np.count_nonzero(x == 12) + np.count_nonzero(x == 13)) \
                * grid_lenght**2 * 25 - vegetable_demand)/6587655 - error_buffer
        g3 = ((np.count_nonzero(x == 1) + np.count_nonzero(x == 2) + np.count_nonzero(x == 12) \
                + np.count_nonzero(x == 13) + np.count_nonzero(x == 14) + np.count_nonzero(x == 15)) \
              * grid_lenght**2 - agri_land_availability)/514949 - error_buffer
        g4 = spatial_adjacency(x)/167796 - error_buffer
        g5 = ((np.count_nonzero(x == 7) + np.count_nonzero(x == 9) + np.count_nonzero(x == 10) \
                + np.count_nonzero(x == 15) + np.count_nonzero(x == 17) + np.count_nonzero(x == 18) \
                    + np.count_nonzero(x == 19)) * grid_lenght**2 - energy_land_availabilty)/404256 - error_buffer
        g6 = -(((((np.count_nonzero(x == 7) + np.count_nonzero(x == 17) + np.count_nonzero(x == 18) + np.count_nonzero(x == 19)) * 0.017) \
                + ((np.count_nonzero(x == 9) + np.count_nonzero(x == 15)) * 0.41) + (np.count_nonzero(x == 10) * 0.014)) \
                    * grid_lenght**2 * 24* 365) - electricity_demand)/654105839 - error_buffer

        out["F"] = [f1, f2]
        out["G"] = [g1, g2, g3, g4, g5, g6]


problem = MyProblem()

algorithm = NSGA2(pop_size=100,
                  sampling=get_sampling("int_random"),
                  crossover=get_crossover("int_sbx"),
                  mutation=get_mutation("int_pm"),
                  eliminate_duplicates=True)

termination = MultiObjectiveDefaultTermination(
    x_tol=1e-8,
    cv_tol=1e-6,
    f_tol=0.0025,
    nth_gen=5,
    n_last=30,
    n_max_gen=3000,
    n_max_evals=300000
)

res = minimize(problem,
               algorithm,
               termination,#=("n_gen", 1000),
               verbose=True,
               seed=1,
               save_history=True)

# Result file name
file_name = 'NSGA_II_test run_g1-3-4-5 pop%s gen%s errorbuffer%s' %(algorithm.pop_size, res.history[-1].n_gen, error_buffer)

# Convergence
n_evals = []    # corresponding number of function evaluations\
F = []          # the objective space values in each generation
G =[]
cv = []         # constraint violation in each generation
# iterate over the deepcopies of algorithms
for algorithm in res.history:

    # store the number of function evaluations
    n_evals.append(algorithm.evaluator.n_eval)

    # retrieve the optimum from the algorithm
    opt = algorithm.opt

    # store the least contraint violation in this generation
    cv.append(opt.get("CV").min())

    # filter out only the feasible and append
    # feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F") #[feas]
    F.append(_F)
    
    G.append(opt.get("G"))


# get the pareto-set and pareto-front for plotting
pf = problem.pareto_front(use_cache=False, flatten=False)

# Decision-making using Decomposition
# weights = np.array([0.5, 0.5])
# decomp = get_decomposition("asf")
# F = problem.pareto_front()
# I = get_decomposition("asf").do(F, weights).argmin()
# print("Best regarding decomposition: Point %s - %s" % (I, F[I]))

# Objective Space
if res.F is None:
    res_F = np.concatenate(F)
else:
    res_F = res.F
min_axis_x = min(res_F[:,0])
min_axis_y = min(res_F[:,1])
best_F = np.argmin((res_F[:,0] - min_axis_x)**2 + (res_F[:,1] - min_axis_y)**2)
plt.style.use('ggplot')
font = FontProperties()
font.set_style('italic')
fig = plt.figure()
plt.scatter(res_F[:,0], res_F[:,1], color="blue")
plt.scatter(res_F[best_F][0], res_F[best_F][1], color="red", marker = '*', s = 90)
plt.title("Objective Space", fontsize = 'medium')
fig.get_axes()[0].set_xlabel("f\u2081", fontsize = 'small', fontproperties = font)
fig.get_axes()[0].set_ylabel("f\u2082", fontsize = 'small', fontproperties = font)
fig.get_axes()[0].tick_params(labelsize = 'small')
plt.show()
png_path = r'C:\Users\20180742\OneDrive - TU Eindhoven\Collaboration\Papers\04-Decision Support System\MOO algorithm scripts\Run Test\%s_Objective space.pdf' %file_name
fig.savefig(png_path, bbox_inches='tight', dpi=300)

# print("Best solution found: %s" % res.X)

#Count res_X
# res_X_count = pd.Series(res.X[best_F]).value_counts()

# Landuse plot
# res_X = res.X[0].reshape(landuse2d.shape)
# values = np.unique(landuse2d.ravel())
# landuse_dict_reverse = {v:k for k,v in landuse_dict.items()}
# im = plt.imshow(landuse2d, interpolation='none')
# colors = [ im.cmap(im.norm(value)) for value in values]
# patches = [ mpatches.Patch(color=colors[i], label=landuse_dict_reverse[i]) for i in range(len(values)) ]
# plt.legend(bbox_to_anchor=(1.05,1), handles= patches, borderaxespad=0., loc=2)
# for i in range(res_X.shape[0]):
#     for j in range(res_X.shape[1]):
#         plt.annotate(res_X[i,j], (i,j))
# plt.show()

# Landuse plot vs. assigned policies
if res.X is None: 
    res_X = lu.make_2d(res.history[-1].opt[0].X)
else: 
    res_X = lu.make_2d(res.X[best_F])
res_X_vect = np.vectorize(lambda x: int(x) if not int(x) in [-1,11] else '')
res_X = res_X_vect(res_X)
values = np.unique(landuse2d.ravel())
landuse_dict_reverse = {v:k for k,v in landuse_dict.items()}
fig = plt.figure()
im = plt.imshow(landuse2d, interpolation='none')
labels = list(landuse_dict.keys())
colors = [ im.cmap(im.norm(value)) for value in values]
patches = [ mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(values)) ]
plt.legend(bbox_to_anchor=(1.05,1), handles= patches, borderaxespad=0., loc=2, prop={'size':6})
for i in range(res_X.shape[0]):
    for j in range(res_X.shape[1]):
        plt.annotate(res_X[i,j], (j,i), xytext = (j-0.25,i+0.25), fontsize=3)
plt.axis('off')
plt.title('Landuse vs. Assigned Policies', fontsize=9)
plt.show()
png_path = r'C:\Users\20180742\OneDrive - TU Eindhoven\Collaboration\Papers\04-Decision Support System\MOO algorithm scripts\Run Test\%s_landuse-vs-policy.pdf' %file_name
fig.savefig(png_path, bbox_inches='tight', dpi=300)


# Get G
if res.G is None:
    res_G = np.concatenate(G)
else:
    res_G = res.G[best_F]

# Plot G
# Gs = np.concatenate(G)
Gs = np.concatenate([x[0] for x in G]).reshape(len(G), problem.n_constr)
plt.style.use('ggplot')
Gs = pd.DataFrame(Gs)
Gs.columns = [f'G{x+1}' for x in range(problem.n_constr)]
# Gs.columns = ['G1', 'G3', 'G4', 'G5']
ax = Gs.plot(linewidth=.5)
plt.title("Constraints", fontsize = 'medium')
ax.set_xlabel("Generation", fontsize = 'small')
ax.set_ylabel("Constraint value", fontsize = 'small')
plt.legend(bbox_to_anchor=(1.05,1))
plt.show()
p = os.path.join(directory, r'%s_G.pdf' %file_name)
ax.get_figure().savefig(p, dpi=300, bbox_inches = 'tight')


# Save Results
if res.X is None: 
    res_X = res.history[-1].opt[0].X
    res_X_best = res_X
else: 
    res_X = res.X
    res_X_best = res_X[best_F]
with pd.ExcelWriter(os.path.join(directory, r'%s.xlsx' %file_name), engine= 'xlsxwriter') as writer:
    pd.DataFrame(res_F).to_excel(writer, sheet_name='res_F', index=False, header=False)
    pd.DataFrame(res_X).to_excel(writer, sheet_name='res_X', index=False, header=False)
    pd.DataFrame(res_G).to_excel(writer, sheet_name='res_G', index=False, header=False)
    pd.DataFrame(lu.make_2d(res_X_best)).to_excel(writer, sheet_name='res_X_Best', index=False, header=False)
    pd.DataFrame(landuse2d).to_excel(writer, sheet_name='landuse', index=False, header=False)


# Hypervolume
ref_point = np.array([0, 0])
metric = Hypervolume(ref_point=ref_point, normalize=False)
hv = [metric.calc(f) for f in F]
plt.plot(n_evals, hv, '-o', markersize=4, linewidth=2)
plt.title("Convergence")
plt.xlabel("Function Evaluations")
plt.ylabel("Hypervolume")
plt.show()

# IGD (Inverted Generational Distance)
pf = problem.pareto_front(use_cache=False, flatten=False)

if pf is not None:

    # for this test problem no normalization for post prcessing is needed since similar scales
    normalize = False

    metric = IGD(pf=pf, normalize=normalize)

    # calculate for each generation the HV metric
    igd = [metric.calc(f) for f in F]

    # visualze the convergence curve
    plt.plot(n_evals, igd, '-o', markersize=4, linewidth=2, color="green")
    plt.yscale("log")          # enable log scale if desired
    plt.title("Convergence")
    plt.xlabel("Function Evaluations")
    plt.ylabel("IGD")
    plt.show()