#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 15:40:16 2019

@author: tao
"""

import torch
import torch.nn as nn


def up(x): 
    return nn.functional.interpolate(x,scale_factor=2)
        
def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool

def conv_decod_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model



class MixedFusion_Block(nn.Module):
    
    def __init__(self,in_dim, out_dim,act_fn):
        super(MixedFusion_Block, self).__init__()
        
        self.layer1 = nn.Sequential(nn.Conv2d(in_dim*3, in_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(in_dim),act_fn,)
        
        # revised in 09/09/2019.
        #self.layer1 = nn.Sequential(nn.Conv2d(in_dim*3, in_dim,  kernel_size=1),nn.BatchNorm2d(in_dim),act_fn,)
        self.layer2 = nn.Sequential(nn.Conv2d(in_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)


    def forward(self, x1,x2,xx):
        
        # multi-style fusion
        fusion_sum = torch.add(x1, x2)   # sum
        fusion_mul = torch.mul(x1, x2)
         
        modal_in1  = torch.reshape(x1,[x1.shape[0],1,x1.shape[1],x1.shape[2],x1.shape[3]])
        modal_in2  = torch.reshape(x2,[x2.shape[0],1,x2.shape[1],x2.shape[2],x2.shape[3]])
        modal_cat  = torch.cat((modal_in1, modal_in2),dim=1)
        fusion_max = modal_cat.max(dim=1)[0]
         
        out_fusion = torch.cat((fusion_sum,fusion_mul,fusion_max),dim=1)
        
        out1 = self.layer1(out_fusion)
        out2 = self.layer2(torch.cat((out1,xx),dim=1))
        
        return out2
        

class MixedFusion_Block0(nn.Module):
    def __init__(self,in_dim, out_dim,act_fn):
        super(MixedFusion_Block0, self).__init__()
        
        self.layer1 = nn.Sequential(nn.Conv2d(in_dim*3, in_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(in_dim),act_fn,)
        #self.layer1 = nn.Sequential(nn.Conv2d(in_dim*3, in_dim, kernel_size=1),nn.BatchNorm2d(in_dim),act_fn,)
        self.layer2 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)


    def forward(self, x1,x2):
        
        # multi-style fusion
        fusion_sum = torch.add(x1, x2)   # sum
        fusion_mul = torch.mul(x1, x2)
         
        modal_in1  = torch.reshape(x1,[x1.shape[0],1,x1.shape[1],x1.shape[2],x1.shape[3]])
        modal_in2  = torch.reshape(x2,[x2.shape[0],1,x2.shape[1],x2.shape[2],x2.shape[3]])
        modal_cat  = torch.cat((modal_in1, modal_in2),dim=1)
        fusion_max = modal_cat.max(dim=1)[0]
         
        out_fusion = torch.cat((fusion_sum,fusion_mul,fusion_max),dim=1)
        
        out1 = self.layer1(out_fusion)
        out2 = self.layer2(out1)
         
        return out2


##############################################
# define our model
class Multi_modal_generator(nn.Module):

    def __init__(self,input_nc, output_nc, ngf):
        super(Multi_modal_generator,self).__init__()
        

        self.in_dim = input_nc
        self.out_dim = ngf
        self.final_out_dim = output_nc
        
        act_fn = nn.LeakyReLU(0.2, inplace=True)
        #act_fn = nn.ReLU()
        
        act_fn2 = nn.ReLU(inplace=True) #nn.ReLU()

        # ~~~ Encoding Paths ~~~~~~ #
        # Encoder (Modality 1)
        
        #######################################################################
        # Encoder **Modality 1
        #######################################################################
        self.down_1_0 = nn.Sequential(
                nn.Conv2d(in_channels=self.in_dim,  out_channels=self.out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.out_dim), act_fn,
                nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.out_dim), act_fn,
                )
        self.pool_1_0 = maxpool()
        
        self.down_2_0 = nn.Sequential(
                nn.Conv2d(in_channels=self.out_dim,  out_channels=self.out_dim*2, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.out_dim*2), act_fn,
                nn.Conv2d(in_channels=self.out_dim*2,out_channels=self.out_dim*2, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.out_dim*2), act_fn,
                )
        self.pool_2_0 = maxpool()
        
        self.down_3_0 = nn.Sequential(
                nn.Conv2d(in_channels=self.out_dim*2,  out_channels=self.out_dim*4, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.out_dim*4), act_fn,
                nn.Conv2d(in_channels=self.out_dim*4,out_channels=self.out_dim*4, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.out_dim*4), act_fn,
                )
        self.pool_3_0 = maxpool()

        
        #######################################################################
        # Encoder **Modality 2
        #######################################################################
        self.down_1_1 = nn.Sequential(
                nn.Conv2d(in_channels=self.in_dim,  out_channels=self.out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.out_dim), act_fn,
                nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.out_dim), act_fn,
                )
        self.pool_1_1 = maxpool()
        
        self.down_2_1 = nn.Sequential(
                nn.Conv2d(in_channels=self.out_dim,  out_channels=self.out_dim*2, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.out_dim*2), act_fn,
                nn.Conv2d(in_channels=self.out_dim*2,out_channels=self.out_dim*2, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.out_dim*2), act_fn,
                )
        self.pool_2_1 = maxpool()
        
        self.down_3_1 = nn.Sequential(
                nn.Conv2d(in_channels=self.out_dim*2,  out_channels=self.out_dim*4, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.out_dim*4), act_fn,
                nn.Conv2d(in_channels=self.out_dim*4,out_channels=self.out_dim*4, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.out_dim*4), act_fn,
                )
        self.pool_3_1 = maxpool()

         
        #######################################################################
        # fusion layer
        #######################################################################
        # down 1st layer
        self.down_fu_1 = MixedFusion_Block0(self.out_dim,self.out_dim*2,act_fn)
        self.pool_fu_1 = maxpool()
        
        self.down_fu_2 = MixedFusion_Block(self.out_dim*2,self.out_dim*4,act_fn)
        self.pool_fu_2 = maxpool()
        
        self.down_fu_3 = MixedFusion_Block(self.out_dim*4,self.out_dim*4,act_fn)
        self.pool_fu_3 = maxpool()

        # down 4th layer
        self.down_fu_4 = nn.Sequential(nn.Conv2d(in_channels=self.out_dim*4,  out_channels=self.out_dim*8, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.out_dim*8), act_fn,)     
     
        # ~~~ Decoding Path ~~~~~~ #
        self.deconv_1_0 = conv_decod_block(self.out_dim * 8, self.out_dim * 4, act_fn2)
        self.deconv_2_0 = MixedFusion_Block(self.out_dim * 4, self.out_dim * 2,act_fn2)
        self.deconv_3_0 = MixedFusion_Block(self.out_dim * 2, self.out_dim * 1,act_fn2)
        self.deconv_4_0 = MixedFusion_Block(self.out_dim * 1, self.out_dim,act_fn2)  
        self.deconv_5_0 = conv_decod_block(self.out_dim * 1, self.out_dim,act_fn2) 
        self.out        = nn.Sequential(nn.Conv2d(int(self.out_dim),1, kernel_size=3, stride=1, padding=1),nn.Tanh()) #  self.final_out_dim
        
        # Modality 1
        self.deconv_1_1 = conv_decod_block(self.out_dim * 4, self.out_dim * 4, act_fn2)
        self.deconv_2_1 = conv_decod_block(self.out_dim * 4, self.out_dim * 4, act_fn2)
        self.deconv_3_1 = conv_decod_block(self.out_dim * 4, self.out_dim * 2, act_fn2)
        self.deconv_4_1 = conv_decod_block(self.out_dim * 2, self.out_dim * 2, act_fn2)
        self.deconv_5_1 = conv_decod_block(self.out_dim * 2, self.out_dim * 1, act_fn2)
        self.deconv_6_1 = conv_decod_block(self.out_dim * 1, int(self.out_dim), act_fn2)
        self.out1       = nn.Sequential(nn.Conv2d(int(self.out_dim),1, kernel_size=3, stride=1, padding=1),nn.Tanh()) #  self.final_out_dim
        
        # modality 2
        self.deconv_1_2 = conv_decod_block(self.out_dim * 4, self.out_dim * 4, act_fn2)
        self.deconv_2_2 = conv_decod_block(self.out_dim * 4, self.out_dim * 4, act_fn2)        
        self.deconv_3_2 = conv_decod_block(self.out_dim * 4, self.out_dim * 2, act_fn2)
        self.deconv_4_2 = conv_decod_block(self.out_dim * 2, self.out_dim * 2, act_fn2)
        self.deconv_5_2 = conv_decod_block(self.out_dim * 2, self.out_dim * 1, act_fn2)
        self.deconv_6_2 = conv_decod_block(self.out_dim * 1, int(self.out_dim), act_fn2)
        self.out2       = nn.Sequential(nn.Conv2d(int(self.out_dim),1, kernel_size=3, stride=1, padding=1),nn.Tanh()) #  self.final_out_dim
    
                
    def forward(self,inputs):

        # ############################# #
        i0 = inputs[:,0:1,:,:]
        i1 = inputs[:,1:2,:,:]
        
        # -----  First Level -------- 
        down_1_0 = self.down_1_0(i0) 
        down_1_1 = self.down_1_1(i1) 


        # -----  Second Level --------
        #input_2nd = torch.cat((down_1_0,down_1_1,down_1_2,down_1_3),dim=1)
        # Max-pool
        down_1_0m   = self.pool_1_0(down_1_0)
        down_1_1m   = self.pool_1_1(down_1_1)
        
        down_2_0 = self.down_2_0(down_1_0m)
        down_2_1 = self.down_2_1(down_1_1m)
        
        
        # -----  Third Level --------
        # Max-pool
        down_2_0m = self.pool_2_0(down_2_0)
        down_2_1m = self.pool_2_1(down_2_1)
                
        down_3_0 = self.down_3_0(down_2_0m)
        down_3_1 = self.down_3_1(down_2_1m)
        
        # Max-pool
        down_3_0m = self.pool_3_0(down_3_0)
        down_3_1m = self.pool_3_1(down_3_1)
         
        # ----------------------------------------
        # fusion layer
        down_fu_1   = self.down_fu_1(down_1_0m,down_1_1m)                                                                                                         
        down_fu_1m  = self.pool_fu_1(down_fu_1)
        
        down_fu_2   = self.down_fu_2(down_2_0m,down_2_1m,down_fu_1m)                                                                                                         
        down_fu_2m  = self.pool_fu_2(down_fu_2)
        
        down_fu_3   = self.down_fu_3(down_3_0m,down_3_1m,down_fu_2m) 
        down_fu_4   = self.down_fu_4(down_fu_3)
        
        #latents     = self.down_fu_4(output_atten)

        #######################################################################                                                                                                
        # ~~~~~~ Decoding 
        deconv_1_0 = self.deconv_1_0(down_fu_4)
        deconv_2_0 = self.deconv_2_0(down_3_0m,down_3_1m,deconv_1_0)
        deconv_3_0 = self.deconv_3_0(down_2_0m,down_2_1m,up(deconv_2_0))
        deconv_4_0 = self.deconv_4_0(down_1_0m,down_1_1m,up(deconv_3_0))
        deconv_5_0 = self.deconv_5_0(up(deconv_4_0))
        output     = self.out(deconv_5_0)

        # modality 1
        deconv_1_1 = self.deconv_1_1((down_3_0m))
        deconv_2_1 = self.deconv_2_1(up(deconv_1_1))
        deconv_3_1 = self.deconv_3_1((deconv_2_1))
        deconv_4_1 = self.deconv_4_1(up(deconv_3_1))
        deconv_5_1 = self.deconv_5_1((deconv_4_1))
        deconv_6_1 = self.deconv_6_1(up(deconv_5_1))
        output1    = self.out(deconv_6_1)
        
        # modality 2
        deconv_1_2 = self.deconv_1_2((down_3_1m))
        deconv_2_2 = self.deconv_2_2(up(deconv_1_2))
        deconv_3_2 = self.deconv_3_2((deconv_2_2))
        deconv_4_2 = self.deconv_4_2(up(deconv_3_2))
        deconv_5_2 = self.deconv_5_2((deconv_4_2))
        deconv_6_2 = self.deconv_6_2(up(deconv_5_2))
        output2    = self.out(deconv_6_2)      
                        
        return output,output1,output2
 

class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()

        def discrimintor_block(in_features, out_features, normalize=True):
            """Discriminator block"""
            layers = [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_features, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discrimintor_block(in_channels, 32, normalize=False),
            *discrimintor_block(32, 64),
            *discrimintor_block(64, 128),
            *discrimintor_block(128, 256),
            #nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, kernel_size=3)
        )

    def forward(self, img):
        return self.model(img)    



  
