# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 19:02:56 2021

@author: Maryam Ghodsvali
"""
import matplotlib.pyplot as plt
# import os
import matplotlib.patches as mpatches
import numpy as np
import os
import pandas as pd
# import rasterio
from matplotlib.font_manager import FontProperties
# from itertools import product
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.problem import Problem
from pymoo.util.termination.default import MultiObjectiveDefaultTermination
from pymoo.optimize import minimize
from pymoo.performance_indicator.hv import Hypervolume
# from pymoo.visualization.scatter import Scatter
from pymoo.factory import  get_sampling, get_crossover, get_mutation#, get_decomposition
# from pymoo.performance_indicator.hv import Hypervolume
# from pymoo.performance_indicator.igd import IGD
from configs import model_vars as v
from set_path import path, sheets
from consts_objs import consts_objs
import math
    
class FWE(Problem):
    def __init__(self):
        super().__init__(n_var=clc.landuse.shape[0],
                         n_obj=2,
                         n_constr=6,
                         xl=np.array([1]*clc.landuse.shape[0]),
                         xu=np.array([1]*clc.landuse.shape[0])*max(clc.cmb.trans.keys()),
                         type_var=int,
                         elementwise_evaluation=True)

    def _evaluate(self, x, out, *args, **kwargs):
        x_trans = clc.translator(x)
        f1 = clc.F1(x_trans)
        f2 = clc.F2(x_trans)
        g1 = clc.G1(x_trans) * 2
        g2 = clc.G2(x_trans)
        g3 = clc.G3(x_trans)
        g4 = clc.G4(x_trans)
        g5 = clc.G5(x_trans)
        g6 = clc.G6(x_trans)

        out["F"] = [f1, f2]
        out["G"] = [g1, g2, g3, g4, g5, g6]
        
class model:
    def __init__(self, scenario = 0, pop_size = 100, norm = False, filename_suffix = ''):
        self.pop_size = pop_size
        self.filename_suffix = filename_suffix
        global clc 
        clc = consts_objs(scenario, norm = norm)
        self.clc = clc
    
    def run(self):
        self.problem = FWE()
        algorithm = NSGA2(pop_size=self.pop_size,
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
        
        res = minimize(
            self.problem,
            algorithm,
            termination,# = ("n_gen", 100),
            verbose=True,
            seed=1,
            save_history=True
            )
        
        self.res = res
        print('Gathering model parameters...')
        self.get_params()
        self.define_file_name()
        self.get_res_F()
        self.get_best_F()
        self.get_res_X()
        self.get_res_G()
        
    def get_params(self):
        n_evals = []    # corresponding number of function evaluations\
        F = []          # the objective space values in each generation
        F_feas = []
        G =[]
        cv = []         # constraint violation in each generation
        # iterate over the deepcopies of algorithms
        for algorithm in self.res.history:
        
            # store the number of function evaluations
            n_evals.append(algorithm.evaluator.n_eval)
            
            # retrieve the optimum from the algorithm
            opt = algorithm.opt
            
            # store the least contraint violation in this generation
            cv.append(opt.get("CV").min())
            
            # filter out only the feasible and append
            feas = np.where(opt.get("feasible"))[0]
            _F_feas = opt.get("F")[feas]
            F_feas.append(_F_feas)
            _F = opt.get("F") #[feas]
            F.append(_F)
            G.append(opt.get("G"))
        
        self.n_evals = n_evals
        self.F = F
        self.F_feas = [x for x in F_feas if not x.shape[0]==0]
        self.G = G
        self.cv = cv
    
    def get_res_F(self):
        if self.res.F is None:
            print('Warning: res.F is None')
            self.res_F = np.concatenate(self.F)
        else:
            self.res_F = self.res.F
            
    def get_best_F(self):
        min_axis_x = min(self.res_F[:,0])
        min_axis_y = min(self.res_F[:,1])
        self.best_F = np.argmin((self.res_F[:,0] - min_axis_x)**2 + (self.res_F[:,1] - min_axis_y)**2)        
    
    def get_res_X(self):
        if self.res.X is None: 
            print('Warning: res.X is None')
            self.res_X = self.clc.lu.make_2d(self.clc.translator(self.res.history[-1].opt[0].X))
            self.res_X_all = self.clc.translator(self.res.history[-1].opt[0].X)
            self.res_X_best = self.res_X_all
        else: 
            self.res_X = self.clc.lu.make_2d(self.clc.translator(self.res.X[self.best_F]))
            self.res_X_all = self.clc.translator(self.res.X)
            self.res_X_best = self.clc.translator(self.res.X[self.best_F])
            
    def get_res_G(self):
        if self.res.G is None:
            print('Warning: res.G is None')
            self.res_G = np.concatenate(self.G)
        else:
            self.res_G = self.res.G[self.best_F]
            
    def define_file_name(self):
        base_name = f'NSGA_II_{self.clc.cmb.scenario}_pop{self.pop_size}_gen{self.res.history[-1].n_gen}_errorbuffer{v.error_buffer}{self.filename_suffix}'
        folder = base_name
        i = 1
        while os.path.isdir(os.path.join(path.path_to_output, folder)):
            folder = folder+'_%d' %i
            i += 1
        os.makedirs(os.path.join(path.path_to_output, folder))
        self.filename = os.path.join(folder, 'out')
        
    def print_out(self, filename):
        print(f'file "{filename}" has been saved')
    
    def get_objective_space(self):
        res_F = self.res_F
        plt.style.use('ggplot')
        font = FontProperties()
        font.set_style('italic')
        fig = plt.figure()
        plt.scatter(res_F[:,0], res_F[:,1], color="blue")
        plt.scatter(res_F[self.best_F][0], res_F[self.best_F][1], color="red", marker = '*', s = 90)
        plt.title("Objective Space", fontsize = 'medium')
        plt.axis('equal')
        fig.get_axes()[0].set_xlabel("f\u2081", fontsize = 'small', fontproperties = font)
        fig.get_axes()[0].set_ylabel("f\u2082", fontsize = 'small', fontproperties = font)
        fig.get_axes()[0].tick_params(labelsize = 'small')
        plt.show()
        file = '%s_Objective space.pdf' %self.filename
        png_path = os.path.join(path.path_to_output, file)
        fig.savefig(png_path, bbox_inches='tight', dpi=300)
        self.print_out(file)
        
    def get_policy_map(self):
        res_X_vect = np.vectorize(lambda x: int(x) if not int(x) in [-1,11] else '')
        res_X = res_X_vect(self.res_X)
        values = np.unique(self.clc.lu.landuse2d.ravel())
        # landuse_dict_reverse = {v:k for k,v in clc.lu.landuse_dict.items()}
        fig = plt.figure()
        im = plt.imshow(self.clc.lu.landuse2d, interpolation='none')
        labels = list(self.clc.lu.landuse_dict.keys())
        colors = [im.cmap(im.norm(value)) for value in values]
        patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(values))]
        plt.legend(bbox_to_anchor=(1.05,1), handles= patches, borderaxespad=0., loc=2, prop={'size':6})
        for i in range(res_X.shape[0]):
            for j in range(res_X.shape[1]):
                plt.annotate(res_X[i,j], (j,i), xytext = (j-0.25,i+0.25), fontsize=3)
        plt.axis('off')
        plt.title('Landuse vs. Assigned Policies', fontsize=9)
        plt.show()
        file = '%s_landuse-vs-policy.pdf' %self.filename
        png_path = os.path.join(path.path_to_output, file)
        fig.savefig(png_path, bbox_inches='tight', dpi=300)
        self.print_out(file)
        
    def get_G_plot(self):
        Gs = np.concatenate([x[0] for x in self.G]).reshape(len(self.G), self.problem.n_constr)
        plt.style.use('ggplot')
        Gs = pd.DataFrame(Gs)
        Gs.columns = [f'G{x+1}' for x in range(self.problem.n_constr)]
        # Gs.columns = ['G1', 'G3', 'G4', 'G5']
        ax = Gs.plot(linewidth=.5)
        plt.title("Constraints", fontsize = 'medium')
        ax.set_xlabel("Generation", fontsize = 'small')
        ax.set_ylabel("Constraint value", fontsize = 'small')
        plt.legend(bbox_to_anchor=(1.05,1))
        plt.show()
        file = '%s_G.pdf' %self.filename
        png_path = os.path.join(path.path_to_output, file)
        ax.get_figure().savefig(png_path, dpi=300, bbox_inches = 'tight')
        self.print_out(file)
        
    def get_hypervolume(self):
        # MODIFY - this is problem dependend
        mx = math.ceil(np.max(np.concatenate(self.F_feas))*10)/10
        ref_point = np.array([mx, mx])
        # create the performance indicator object with reference point
        metric = Hypervolume(ref_point=ref_point, normalize=False)
        
        # calculate for each generation the HV metric
        hv = [metric.calc(f) for f in self.F_feas]
        
        # visualze the convergence curve
        fig = plt.figure()
        plt.style.use('ggplot')
        plt.plot(self.n_evals[-len(self.F_feas):], hv, '-o', markersize=4, linewidth=2)
        plt.title("Convergence")
        plt.xlabel("Function Evaluations")
        plt.ylabel("Hypervolume")
        plt.show()
        file = '%s_hypervolume.pdf' %self.filename
        png_path = os.path.join(path.path_to_output, file)
        fig.savefig(png_path, dpi=300, bbox_inches = 'tight')
        self.print_out(file)
        
    def save_results(self):
        file = os.path.join(path.path_to_output, r'%s.xlsx' %self.filename)
        with pd.ExcelWriter(file, engine= 'xlsxwriter') as writer:
            pd.DataFrame(self.res_F).to_excel(writer, sheet_name=sheets.res_f, index=False, header=False)
            pd.DataFrame(self.res_X_all).to_excel(writer, sheet_name=sheets.res_x_all, index=False, header=False)
            pd.DataFrame(self.res_G).to_excel(writer, sheet_name=sheets.res_g, index=False, header=False)
            pd.DataFrame(self.clc.lu.make_2d(self.res_X_best)).to_excel(writer, sheet_name=sheets.res_x_best, index=False, header=False)
            pd.DataFrame(self.clc.lu.landuse2d).to_excel(writer, sheet_name=sheets.landuse, index=False, header=False)
        self.print_out(file)