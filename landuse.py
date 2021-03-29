# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 19:02:56 2021

@author: Maryam Ghodsvali
"""
import rasterio
import numpy as np
from set_path import path

class landuse_creator:
    def __init__(self, random_proportion = [0.57, 0.37, 0.06], exclude = [0,15]):
        self.random_proportion = random_proportion
        self.exclude = exclude
    
    def read2d_partial_random(self):
        p = path.path_to_landuse
        raster = rasterio.open(p)
        raster = raster.read(1)
        val_convert = np.vectorize(lambda x: -1 if x in self.exclude else x)
        raster = val_convert(raster)
        raster_flat = raster.reshape(raster.shape[0]*raster.shape[1])
        mask = raster_flat==1
        other = np.random.choice(len(self.random_proportion), np.sum(mask), p=self.random_proportion)
        raster_flat[mask] = other
        self.landuse2d = raster
        return raster
    
    def read(self):
        self.landuse_size = (self.landuse2d.shape[0], self.landuse2d.shape[1])
        self.landuse_mask = self.landuse2d!=-1
        self.landuse = self.landuse2d.reshape(self.landuse2d.shape[0]*self.landuse2d.shape[1])
        self.landuse = self.landuse[self.landuse!=-1]
        rev = np.vectorize(lambda x:not x)
        self.landuse_mask_rev = rev(self.landuse_mask)
        return self.landuse
    
    def get_size(self):
        return self.landuse_size

    def make_2d(self, landuse):
        landuse2d = np.zeros(self.landuse_size)
        landuse2d[self.landuse_mask] = landuse
        landuse2d[self.landuse_mask_rev] = -1
        return landuse2d