#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tus April 18 17:18:50 2019

@author: tao
"""
#coding:utf8
import os
from torch.utils import data
import numpy as np
from torchvision import  transforms as T
from funcs.utils import *
import torch
import scipy.io as scio

  
def loadSubjectData(path):
    
    data_imgs = scio.loadmat(path) 
    
    img_flair = data_imgs['data_img']['flair'][0][0].astype(np.float32)
    img_t1    = data_imgs['data_img']['t1'][0][0].astype(np.float32)
    img_t1ce  = data_imgs['data_img']['t1ce'][0][0].astype(np.float32)
    img_t2    = data_imgs['data_img']['t2'][0][0].astype(np.float32)
    
    # crop 160*180 images
    img_t1    = img_t1[40:200,20:200,:]
    img_t1ce  = img_t1ce[40:200,20:200,:]
    img_t2    = img_t2[40:200,20:200,:]
    img_flair = img_flair[40:200,20:200,:]
               
    return img_t1,img_t1ce,img_t2,img_flair


class MultiModalityData_load(data.Dataset):
    
    def __init__(self,opt,transforms=None,train=True,test=False):
        
        self.opt   = opt
        self.test  = test
        self.train = train
        
        if self.test:
            path_test   = opt.data_path + 'test/'
            data_paths  = [os.path.join(path_test,i) for i in os.listdir(path_test)] 
            
        if self.train:
            path_train  = opt.data_path + 'train/'
            data_paths  = [os.path.join(path_train,i) for i in os.listdir(path_train)] 
   
        data_paths      = sorted(data_paths,key=lambda x:int(x.split('.')[0].split('_')[-1]))
        self.data_paths = np.array(data_paths)
                
        
    def __getitem__(self,index):
        
        # path
        cur_path  = self.data_paths[index]
        
        # get images
        img_t1,img_t1ce,img_t2,img_flair = loadSubjectData(cur_path)
        
      
        # split into patches (128*128) 
        img_t1_patches    = generate_all_2D_patches(img_t1)
        img_t1ce_patches  = generate_all_2D_patches(img_t1ce)
        img_t2_patches    = generate_all_2D_patches(img_t2)
        img_flair_patches = generate_all_2D_patches(img_flair)

        return img_t1_patches,img_t1ce_patches,img_t2_patches,img_flair_patches
    
    
    
    def __len__(self):
        return len(self.data_paths)
    
     
