#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 10:15:11 2019

@author: tao
"""

#from SurvivalPredictionModel import SurvivalPredictionModel
from HiNet_SynthModel import LatentSynthModel
from config import opt
import fire

    
def train(**kwargs):
    
    opt.parse(kwargs)
    
    SynModel = LatentSynthModel(opt=opt)
    SynModel.train() 
    

def test(**kwargs):
    
    opt.parse(kwargs)
    SynModel = LatentSynthModel(opt=opt)
    SynModel.test(0) 

        
   
if __name__ == '__main__':
    
    fire.Fire()