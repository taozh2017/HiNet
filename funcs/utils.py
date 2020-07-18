#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 11:33:55 2019

@author: tao
"""
import os
import scipy.io as sio 
import numpy as np
import torch
import matplotlib.pyplot as plt
import math
from numpy.lib.stride_tricks import as_strided as ast
f#rom skimage.measure import structural_similarity as ssim
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
         
 
def generate_2D_patches(in_data):
    
    # in_data     --> 240*240*155 for BraTS 
    # out_patch   --> 128*128*128
    # num_patches --> num = 9
    #in_size  = [240,240,155]
    in_size  = [160,180]
    out_size = [128,128]
    num      = 2  
    
    x_locs   = generate_patch_loc(in_size[0],out_size[0],num)
    y_locs   = generate_patch_loc(in_size[1],out_size[1],num)
  
    
    patches  = np.zeros([num*num,out_size[0],out_size[1]])
    count    = 0

    for i in range(len(x_locs)):
        for j in range(len(y_locs)):
            xx = x_locs[i][0]
            yy = y_locs[j][0]
            
            patches[count,:,:] = in_data[xx:xx+out_size[0],yy:yy+out_size[1]]
            count = count + 1
                
    return patches  


def generate_all_2D_patches(in_data):
    
    #in_size  = [160,180]
    used_data   = in_data[:,:]
    out_size    = [128,128]
    num         = 1
    out_patches = np.zeros([num*4, out_size[0],out_size[1]])
    for i in range(num):
        out_patches[i*4:(i+1)*4] = generate_2D_patches(used_data[:,:])
        
    return out_patches


def generate_2D_pathches_slice_test(in_data):
    
    #in_size  = [160,180,155]
    used_data = in_data
    out_size  = [128,128]
    num       = used_data.shape[2]
    out_patches = np.zeros([num*4, out_size[0],out_size[1]])
    for i in range(num):
        out_patches[i*4:(i+1)*4] = generate_2D_patches(used_data[:,:,i])
        
    return out_patches

def generate_2D_patches_slice(in_data):
    
    #in_size  = [160,180]
    used_data = in_data
    out_size  = [128,128]
    num       = 1
    out_patches = np.zeros([num*4, out_size[0],out_size[1]])
    out_patches = generate_2D_patches(used_data[:,:])
        
    return out_patches


def generate_patch_loc(in_size,out_size,num):
    
    locs  = np.zeros([num,1])
    for i in range(num):
        if i == 0: 
            locs[i] = 0
        else:
            locs[i] = int((in_size-out_size)/(num-1))*i 
            
    #locs[i] = in_size-out_size - 1
    
    return locs.astype(int)

def prediction_in_testing_2Dimages(x_in,x03_real):
    
    
    x_in_re = torch.reshape(torch.reshape(x_in,[x_in.shape[0],x_in.shape[2],x_in.shape[3]]), [int(x_in.shape[0]/4),4,x_in.shape[2],x_in.shape[3]])
   
    in_size  = [160,180]
    out_size = [128,128]
    num      = 2  
    x_locs   = generate_patch_loc(in_size[0],out_size[0],num)
    y_locs   = generate_patch_loc(in_size[1],out_size[1],num)
     
    pred_images  = np.zeros([x_in_re.shape[0],in_size[0],in_size[1]])
    pred_values  = np.zeros([x_in_re.shape[0],3])
    
    for k in range(x_in_re.shape[0]):
        
        count  = 0
        matOut = torch.zeros((in_size[0],in_size[1]))
        used   = torch.zeros((in_size[0],in_size[1]))  
        
        cur_real_data = torch.reshape(x03_real[:,:,k],[160,180])
        cur_real_data = cur_real_data - cur_real_data.min()
        cur_real_data = cur_real_data/cur_real_data.max()
                
        for i in range(len(x_locs)):
            for j in range(len(y_locs)):
                xx = x_locs[i][0]
                yy = y_locs[j][0]
                        
                temp_out = x_in_re[k,count,:,:]
                temp_out = torch.reshape(temp_out,[128,128])
                        
                        
                # normalization
                temp_out = temp_out - temp_out.min()
                temp_out = temp_out/temp_out.max()
                        
                        
                matOut[xx:xx+out_size[0],yy:yy+out_size[1]] = matOut[xx:xx+out_size[0],yy:yy+out_size[1]] + temp_out.cpu()
                used[xx:xx+out_size[0],yy:yy+out_size[1]]   = used[xx:xx+out_size[0],yy:yy+out_size[1]] + 1
                
                count = count + 1
                
        #--------------------
        pred_res = matOut/used
        pred_res = pred_res - pred_res.min()
        pred_res = pred_res/pred_res.max()
        
        cur_real_data = cur_real_data - cur_real_data.min()
        cur_real_data = cur_real_data/cur_real_data.max()
        
        # --------------------
        psnr = compute_psnr(pred_res, cur_real_data)
        nmse = compute_nmse(pred_res, cur_real_data)
        ssim = compute_ssim(pred_res, cur_real_data)
                    
        pred_images[k,:,:] = (matOut/used).cpu().detach().numpy()
        pred_values[k,:]   = [psnr,nmse,ssim]
        
    return pred_images,pred_values



def prediction_in_testing_2DimagesNEW(pred_out,x03_real):
    #print(pred_out.shape)
    x_in = pred_out
    x_in_re = torch.reshape(torch.reshape(x_in,[x_in.shape[0],x_in.shape[2],x_in.shape[3]]), [int(x_in.shape[0]/4),4,x_in.shape[2],x_in.shape[3]])
   
    in_size  = [160,180]
    out_size = [128,128]
    num      = 2  
    x_locs   = generate_patch_loc(in_size[0],out_size[0],num)
    y_locs   = generate_patch_loc(in_size[1],out_size[1],num)
     
    pred_images  = np.zeros([x_in_re.shape[0],in_size[0],in_size[1]])
    pred_values  = np.zeros([x_in_re.shape[0],2])
    
    for k in range(x_in_re.shape[0]):
        
        count  = 0
        matOut = torch.zeros((in_size[0],in_size[1]))
        used   = torch.zeros((in_size[0],in_size[1]))  
        
        cur_real_data = torch.reshape(x03_real[:,:,k],[160,180])
        cur_real_data = cur_real_data - cur_real_data.min()
        
        cur_real_data = cur_real_data/cur_real_data.max()
#        print('our model:',[cur_real_data.max(),cur_real_data.min(),aa.max(),aa.min()])
                
        for i in range(len(x_locs)):
            for j in range(len(y_locs)):
                xx = x_locs[i][0]
                yy = y_locs[j][0]
                        
                temp_out = x_in_re[k,count,:,:]
                temp_out = torch.reshape(temp_out,[128,128])
                        
                        
                # normalization
                temp_out = temp_out - temp_out.min()
                temp_out = temp_out/temp_out.max()
                        
                        
                matOut[xx:xx+out_size[0],yy:yy+out_size[1]] = matOut[xx:xx+out_size[0],yy:yy+out_size[1]] + temp_out.cpu()
                used[xx:xx+out_size[0],yy:yy+out_size[1]]   = used[xx:xx+out_size[0],yy:yy+out_size[1]] + 1
                
                count = count + 1
                
        #--------------------
        pred_res = matOut/used +0.2
        pred_res = pred_res - pred_res.min()
        pred_res = pred_res/pred_res.max()
        
#        cur_real_data = cur_real_data - cur_real_data.min()
#        cur_real_data = cur_real_data/cur_real_data.max()
        
        # --------------------
        psnr = compute_psnr(pred_res, cur_real_data)
        nmse = compute_nmse(pred_res, cur_real_data)
       # ssims = compute_ssim(pred_res, cur_real_data)
                    
        pred_images[k,:,:] = (matOut/used).cpu().detach().numpy()
        pred_values[k,:]   = [psnr,nmse]
        #pred_values = 1
    return pred_images,pred_values



def prediction_syn_results(pred_out,real_out):
    
    ###########################################################################
    ### Note that
    # there is two manners to evaluate the testing sets
    # for example, using T1 + T2 to synthesize Flair
    # -->(1) the ground truths of Flair keep original size ([160,180,batch_size]) without spliting into small pathces (128*128). In this case, the 
    # synthesized results with size [batch_size*num_patch,1,128,128]， we need change it to [160,180,batch_size]
     
    # -->(2) the ground truths and synthesized results are all with size [batch_size*num_patch,1,128,128]， we need change 
    # them to [160,180,batch_size]. See details of this maner below.
    
    # When one volume as input, we set batch_size=num_slice
        
    ###########################################################################    

    
    # [batch_size*num_patch,1,128,128] -- > [batch_size, num_patch, 128, 128]
    pred_out_re = torch.reshape(torch.reshape(pred_out,[pred_out.shape[0],pred_out.shape[2],pred_out.shape[3]]), [int(pred_out.shape[0]/4),4,pred_out.shape[2],pred_out.shape[3]])
    real_out_re = torch.reshape(torch.reshape(real_out,[real_out.shape[0],real_out.shape[2],real_out.shape[3]]), [int(real_out.shape[0]/4),4,real_out.shape[2],real_out.shape[3]])

    in_size  = [160,180]
    out_size = [128,128]
    num      = 2  # num_patch = num*num
    
    x_locs   = generate_patch_loc(in_size[0],out_size[0],num)
    y_locs   = generate_patch_loc(in_size[1],out_size[1],num)
     
    pred_images  = np.zeros([in_size[0],in_size[1],pred_out_re.shape[0]])
    real_images  = np.zeros([in_size[0],in_size[1],pred_out_re.shape[0]])
    
    for k in range(pred_out_re.shape[0]):
        
        count  = 0
        mat_pred_Out = torch.zeros((in_size[0],in_size[1]))
        used_pred    = torch.zeros((in_size[0],in_size[1]))  
        
        mat_real_Out = torch.zeros((in_size[0],in_size[1]))
        used_real    = torch.zeros((in_size[0],in_size[1]))  
     
        ## 
        for i in range(len(x_locs)):
            for j in range(len(y_locs)):
                xx = x_locs[i][0]
                yy = y_locs[j][0]
                        
                temp_pred_out = pred_out_re[k,count,:,:]
                temp_pred_out = torch.reshape(temp_pred_out,[128,128])
                        
                temp_real_out = real_out_re[k,count,:,:]
                temp_real_out = torch.reshape(temp_real_out,[128,128])
                        
                # normalization
                temp_pred_out = temp_pred_out - temp_pred_out.min()
                temp_pred_out = temp_pred_out/temp_pred_out.max()
                
                temp_real_out = temp_real_out - temp_real_out.min()
                temp_real_out = temp_real_out/temp_real_out.max()
                        
                        
                mat_pred_Out[xx:xx+out_size[0],yy:yy+out_size[1]] = mat_pred_Out[xx:xx+out_size[0],yy:yy+out_size[1]] + temp_pred_out.cpu()
                used_pred[xx:xx+out_size[0],yy:yy+out_size[1]]    = used_pred[xx:xx+out_size[0],yy:yy+out_size[1]] + 1
 
                mat_real_Out[xx:xx+out_size[0],yy:yy+out_size[1]] = mat_real_Out[xx:xx+out_size[0],yy:yy+out_size[1]] + temp_real_out.cpu()
                used_real[xx:xx+out_size[0],yy:yy+out_size[1]]    = used_real[xx:xx+out_size[0],yy:yy+out_size[1]] + 1
                
                
                count = count + 1
                
        #--------------------
        pred_res = mat_pred_Out/used_pred 
        real_res = mat_real_Out/used_real 
        
             
        pred_images[:,:,k] = pred_res.detach().numpy()#pred_res.cpu().detach().numpy()
        real_images[:,:,k] = real_res.detach().numpy() 
        
    
    pred_images = pred_images - pred_images.min()
    pred_images = pred_images/pred_images.max()
    
    real_images = real_images - real_images.min()
    real_images = real_images/real_images.max()  
    
    errors = ErrorMetrics(pred_images.astype(np.float32), real_images.astype(np.float32))  
        
    return errors


def loadSubjectData(path):
    
    data_imgs = sio.loadmat(path) 
    
    img_flair = data_imgs['data']['img_flair'][0][0].astype(np.float32)
    img_t1    = data_imgs['data']['img_t1'][0][0].astype(np.float32)
    img_t1ce  = data_imgs['data']['img_t1ce'][0][0].astype(np.float32)
    img_t2    = data_imgs['data']['img_t2'][0][0].astype(np.float32)
            
    return img_t1,img_t1ce,img_t2,img_flair
 


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)
    

class Logger(object):
	'''Save training process to log file with simple plot function.'''
	def __init__(self, fpath, title=None, resume=False): 
		self.file = None
		self.resume = resume
		self.title = '' if title == None else title
		if fpath is not None:
			if resume: 
				self.file = open(fpath, 'r') 
				name = self.file.readline()
				self.names = name.rstrip().split('\t')
				self.numbers = {}
				for _, name in enumerate(self.names):
					self.numbers[name] = []

				for numbers in self.file:
					numbers = numbers.rstrip().split('\t')
					for i in range(0, len(numbers)):
						self.numbers[self.names[i]].append(numbers[i])
				self.file.close()
				self.file = open(fpath, 'a')  
			else:
				self.file = open(fpath, 'w')

	def set_names(self, names):
		if self.resume: 
			pass
		# initialize numbers as empty list
		self.numbers = {}
		self.names = names
		for _, name in enumerate(self.names):
			self.file.write(name)
			self.file.write('\t')
			self.numbers[name] = []
		self.file.write('\n')
		self.file.flush()


	def append(self, numbers):
		assert len(self.names) == len(numbers), 'Numbers do not match names'
		for index, num in enumerate(numbers):
			self.file.write("{0:.6f}".format(num))
			self.file.write('\t')
			self.numbers[self.names[index]].append(num)
		self.file.write('\n')
		self.file.flush()

	def plot(self, names=None):   
		names = self.names if names == None else names
		numbers = self.numbers
		for _, name in enumerate(names):
			x = np.arange(len(numbers[name]))
			plt.plot(x, np.asarray(numbers[name]))
		plt.legend([self.title + '(' + name + ')' for name in names])
		plt.grid(True)

	def close(self):
		if self.file is not None:
			self.file.close()
             

class AverageMeter(object):
	"""Computes and stores the average and current value
	   Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
	"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		
	def avg(self):
		return self.sum / self.count

def mkdir_p(path):
	'''make dir if not exist'''
	try:
		os.makedirs(path)
	except OSError as exc:  # Python >2.5
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		else:
			raise    
            

        
def model_task(inputs,task_id):
    
    if task_id == 1:
        in_id1 = 0
        in_id2 = 1
        out_id = 2

    if task_id == 2:
        in_id1 = 0
        in_id2 = 2
        out_id = 3 
        
    if task_id == 3:
        in_id1 = 0
        in_id2 = 3
        out_id = 2 
        
    if task_id == 4:
        in_id1 = 2
        in_id2 = 3
        out_id = 0
        
             
    x1 = torch.reshape(inputs[in_id1], [inputs[in_id1].shape[1]*inputs[in_id1].shape[0],1,inputs[in_id1].shape[2],inputs[in_id1].shape[3]]).type(torch.FloatTensor)        
    x2 = torch.reshape(inputs[in_id2], [inputs[in_id2].shape[1]*inputs[in_id2].shape[0],1,inputs[in_id2].shape[2],inputs[in_id2].shape[3]]).type(torch.FloatTensor)
                 
    x3 = torch.reshape(inputs[out_id], [inputs[out_id].shape[1]*inputs[out_id].shape[0],1,inputs[out_id].shape[2],inputs[out_id].shape[3]]).type(torch.FloatTensor)
                     
    return x1,x2,x3

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
        
def ErrorMetrics(vol_s, vol_t):
    
    # calculate various error metrics.
    # vol_s should be the synthesized volume (a 3d numpy array) or an array of these volumes
    # vol_t should be the ground truth volume (a 3d numpy array) or an array of these volumes

#    vol_s = np.squeeze(vol_s)
#    vol_t = np.squeeze(vol_t)
    
#    vol_s = vol_s.numpy()
#    vol_t = vol_t.numpy()

    assert len(vol_s.shape) == len(vol_t.shape) == 3
    assert vol_s.shape[0] == vol_t.shape[0]
    assert vol_s.shape[1] == vol_t.shape[1]
    assert vol_s.shape[2] == vol_t.shape[2]

    vol_s[vol_t == 0] = 0
    vol_s[vol_s < 0] = 0

    errors = {}
    
    vol_s = vol_s.astype(np.float32)
      
    # errors['MSE'] = np.mean((vol_s - vol_t) ** 2.)
    errors['MSE'] = np.sum((vol_s - vol_t) ** 2.) / np.sum(vol_t**2)
    errors['SSIM'] = ssim(vol_t, vol_s)
    dr = np.max([vol_s.max(), vol_t.max()]) - np.min([vol_s.min(), vol_t.min()])
    errors['PSNR'] = psnr(vol_t, vol_s, dynamic_range=dr)

#    # non background in both
#    non_bg = (vol_t != vol_t[0, 0, 0])
#    errors['SSIM_NBG'] = ssim(vol_t[non_bg], vol_s[non_bg])
#    dr = np.max([vol_t[non_bg].max(), vol_s[non_bg].max()]) - np.min([vol_t[non_bg].min(), vol_s[non_bg].min()])
#    errors['PSNR_NBG'] = psnr(vol_t[non_bg], vol_s[non_bg], dynamic_range=dr)
#
#    vol_s_non_bg = vol_s[non_bg].flatten()
#    vol_t_non_bg = vol_t[non_bg].flatten()
#    
#    # errors['MSE_NBG'] = np.mean((vol_s_non_bg - vol_t_non_bg) ** 2.)
#    errors['MSE_NBG'] = np.sum((vol_s_non_bg - vol_t_non_bg) ** 2.) /np.sum(vol_t_non_bg**2)

    return errors
