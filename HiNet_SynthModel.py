#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:04:44 2019

@author: tao
"""

import os
import torch
import torch.nn as nn
import numpy as np

import time
import datetime

from torch.utils.data import DataLoader

#from models import *
#from fusion_models import *  # revise in 09/03/2019
from dataset import MultiModalityData_load
from funcs.utils import *
import torch.nn as nn
import scipy.io as scio
from torch.autograd import Variable
import torch.autograd as autograd
#import IVD_Net as IVD_Net
import model.syn_model as models



#from config import opt
#from visualize import Visualizer
#testing     

#os.environ["CUDA_VISIBLE_DEVICES"] = '5,6'

cuda = True if torch.cuda.is_available() else False
FloatTensor   = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor    = torch.cuda.LongTensor if cuda else torch.LongTensor


class LatentSynthModel():
    
    ########################################################################### 
    
    def __init__(self,opt):
        
        self.opt         = opt  
        self.generator   = models.Multi_modal_generator(1,1,32)
        self.discrimator = models.Discriminator()
        
        if opt.use_gpu: 
            self.generator    = self.generator.cuda()
            self.discrimator  = self.discrimator.cuda()
                        
        if torch.cuda.device_count() > 1:
            self.generator    = nn.DataParallel(self.generator,device_ids=self.opt.gpu_id)
            self.discrimator  = nn.DataParallel(self.discrimator,device_ids=self.opt.gpu_id)  
        


    ########################################################################### 
    def train(self):   
        
        if not os.path.isdir(self.opt.save_path+'/'+'task_'+str(self.opt.task_id)+'/'):
            mkdir_p(self.opt.save_path+'/'+'task_'+str(self.opt.task_id)+'/')
        
        logger = Logger(os.path.join(self.opt.save_path+'/'+'task_'+str(self.opt.task_id)+'/'+'run_log.txt'), title='')
        logger.set_names(['Run epoch', 'D Loss', 'G Loss'])

        #
        self.generator.apply(weights_init_normal)
        self.discrimator.apply(weights_init_normal)
        print('weights_init_normal')
                
        # Optimizers
        optimizer_D     = torch.optim.Adam(self.discrimator.parameters(), lr=self.opt.lr,betas=(self.opt.b1, self.opt.b2))
        optimizer_G     = torch.optim.Adam(self.generator.parameters(),lr=self.opt.lr,betas=(self.opt.b1, self.opt.b2))

        # Learning rate update schedulers
        lr_scheduler_G  = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(self.opt.epochs, 0, self.opt.decay_epoch).step)
        lr_scheduler_D  = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(self.opt.epochs, 0, self.opt.decay_epoch).step)
    
            
        # Lossesgenerator
        criterion_GAN   = nn.MSELoss().cuda()
        criterion_identity = nn.L1Loss().cuda()

        # Load data        
        train_data   = MultiModalityData_load(self.opt,train=True)
        train_loader = DataLoader(train_data,batch_size=self.opt.batch_size,shuffle=False)


        batches_done = 0
        prev_time    = time.time()
        # ---------------------------- *training * ---------------------------------
        for epoch in range(self.opt.epochs):      
            for ii, inputs in enumerate(train_loader):
                print(ii)
                
                # define diferent synthesis tasks
                [x1,x2,x3] = model_task(inputs,self.opt.task_id) # train different synthesis task
                
                fake  = torch.zeros([inputs[0].shape[1]*inputs[0].shape[0],1,6,6], requires_grad=False) #.cuda()
                valid = torch.ones([inputs[0].shape[1]*inputs[0].shape[0],1,6,6], requires_grad=False)#.cuda()
                              
                ###############################################################                     
                if self.opt.use_gpu:
                    x1   = x1.cuda()
                    x2   = x2.cuda()
                    x3   = x3.cuda()
                    
                x_fu = torch.cat([x1,x2],dim=1)

                # ----------------------
                #  Train generator
                # ----------------------
                optimizer_G.zero_grad()
                
                x_fake,x1_re,x2_re = self.generator(x_fu)
                
                # Identity loss
                loss_re3 = criterion_identity(x_fake, x3)
                loss_re1 = criterion_identity(x1_re, x1)
                loss_re2 = criterion_identity(x2_re, x2)
                
  
                # gan loss 
                loss_GAN = criterion_GAN(self.discrimator(x_fake), valid) 
                            
                # total loss
                loss_G = loss_GAN + 100*loss_re3 + 20*loss_re1 + 20*loss_re2
                
            
                loss_G.backward(retain_graph=True)
                optimizer_G.step()
            
                # ----------------------
                #  Train Discriminators
                # ----------------------
                optimizer_D.zero_grad()

                
                # Real loss
                loss_real = criterion_GAN(self.discrimator(x3), valid)
                loss_fake = criterion_GAN(self.discrimator(x_fake), fake)
                # Total loss
                loss_D = (loss_real + loss_fake) / 2
        
                loss_D.backward(retain_graph=True)
                optimizer_D.step()
                
                # time
                batches_left = self.opt.epochs * len(train_loader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time) / self.opt.n_critic)
                prev_time = time.time()
                    
                    #print('Epoch:', epoch, '| D_loss: %.6f' % loss_D.item(),'| G_loss: %.6f' % loss_G.item())
                print('\r[Epoch %d/%d]:' % (epoch, self.opt.epochs),'[Batch %d/%d]:' % (ii, len(train_loader)), '| D_loss: %.6f' % loss_D.item(),'| G_loss: %.6f' % loss_G.item(),'ETA: %s' %time_left)
            
                batches_done += 1
                
   
            # Update learning rates
            lr_scheduler_G.step()
            lr_scheduler_D.step()

            logger.append([epoch, loss_D.item(), loss_G.item()])
            
            # Save model checkpoints
            if epoch > 20 and (epoch) % self.opt.checkpoint_interval == 0:
                
                torch.save(self.generator.state_dict(),  self.opt.save_path+'/'+'task_'+str(self.opt.task_id)+'/generator_%d.pkl' % (epoch))
                torch.save(self.discrimator.state_dict(),self.opt.save_path+'/'+'task_'+str(self.opt.task_id)+'/discrimator_%d.pkl' % (epoch))


    ###########################################################################
    def test(self,ind_epoch):   
         
        self.generator.load_state_dict(torch.load(self.opt.save_path+'/'+'task_'+str(self.opt.task_id)+'/'+ 'generator_'+str(ind_epoch)+'.pkl'),strict=False)   
       
        # Load data        
        te_data   = MultiModalityData_load(self.opt,train=False,test=True)
        te_loader = DataLoader(te_data,batch_size=self.opt.batch_size,shuffle=False)
        
        pred_eva_set = []
        for ii, inputs in enumerate(te_loader): 
            #print(ii) 
            # define diferent synthesis tasks
            [x_in1, x_in2, x_out] = model_task(inputs,self.opt.task_id)
            x_fusion   = torch.cat([x_in1,x_in2],dim=1)
                      
            if self.opt.use_gpu:
                x_fusion     = x_fusion.cuda()
            
            
            # pred_out -- [batch_size*4,1,128,128]
            # x3       -- [batch_size*4,1,128,128]
            pred_out,pred_out1,pred_out2 = self.generator(x_fusion) 


            errors = prediction_syn_results(pred_out,x_out)  
            
            print(errors)
           
            pred_eva_set.append([errors['MSE'],errors['SSIM'],errors['PSNR']])
        
        mean_values = [ind_epoch,np.array(pred_eva_set)[:,0].mean(),np.array(pred_eva_set)[:,1].mean(),np.array(pred_eva_set)[:,2].mean(),np.array(pred_eva_set)[:,3].mean(),np.array(pred_eva_set)[:,4].mean(),np.array(pred_eva_set)[:,5].mean()]

        return mean_values
    
    
    