# TODO this is temporary until we create a subpackage!
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[3]
library_root = root / "library"
for path in (root, library_root):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

import library
thisfiledir = Path(__file__).resolve().parent
expdir = '/home/adminlwe/Documents/lwe-opu/experiment'
savedir = expdir + '/data/model_training'
calibdir = thisfiledir / "calibration"
import time
import numpy as np
import ipywidgets
import pickle
import cv2
from shutil import copyfile
from datetime import datetime

# Pytorch imports
import torch
import torchvision
import torch.nn as nn

import torch.optim as optim
from torch.optim import Adam, RMSprop
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader, default_collate

# Custom modules and NN
from library.software.extra import ComplexLayerNorm, ComplexAvgPool2d, CPU2GPU, DebugFigure, Repeat
from library.software.layers import FFConv, FFLinear, FFElmwise, FFPhaseElmwise
from library.software.layers import Screen, Camera, Mask

from library.software.architectures import NormalNet, FFNet
from library.software.dataloaders import MNIST_loaders
from op_torch import img_nav

# Camera imports
from library.hardware.PylonCamera import PylonCamera
from library.hardware import DispUtils

# Display imports
from library.hardware import DispUtils
from library.hardware.DisplayGL import DisplayGL
from library.hardware.PylonCamera import PylonCamera
from library.hardware.DeviceManager import DeviceManager

# Select GPU as device if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

global calibfile
calibfile = input('Name of calibration/data file ?')
os.chdir(calibdir)
file = open(f'data_{calibfile}', 'rb')
[zoom, img_size, slm_size, pat_pad, cam_pad, batch_layout, batch_stacks, shape_labeled, Npix] = pickle.load(file)
file.close() ; os.chdir(thisfiledir)

Nmux = batch_layout[0]*batch_layout[1]
Npix_x, Npix_y = Npix[0], Npix[1]
Npixtot = Npix_x * Npix_y
Nin = shape_labeled[0]*shape_labeled[1]

def generate_flat_slm_mask(shape_labeled, value=0):
    '''
    Input:  shape_labeled: [Lx,Ly] vector
            n_repeat: nb of px for the same value
    Output: slm_stack with the same random mask of shape [batch_stacks, Nmux, Lx, Ly]     
    '''
    random = torch.from_numpy(value*np.ones([shape_labeled[0],shape_labeled[1]]))
    slm_stack = torch.stack([ torch.stack([ random for i in range(batch_layout[0]*batch_layout[1])])  for i in range(batch_stacks)]).to(torch.device("cpu:0"))
    return slm_stack

def generate_random_slm_mask(shape_labeled, n_repeat=2, same=True):
    '''
    Input:  shape_labeled: [lx,ly] vector
            n_repeat: nb of px for the same value
            same: repetition of the same maks on all the subimages or different masks everywhere
    Output: slm_stack with the same random mask of shape [batch_stacks, Nx, Ny, Lx, Ly]     
    '''
    if same: #we want the same random mask for each subimages
        random = torch.from_numpy(np.random.uniform(size=[shape_labeled[0]//n_repeat,shape_labeled[1]//n_repeat]))
        slm_stack = torch.repeat_interleave(random, n_repeat, dim = -1)
        slm_stack = torch.repeat_interleave(slm_stack, n_repeat, dim = -2)
        slm_stack = torch.stack([torch.stack([ slm_stack for i in range(batch_layout[0]*batch_layout[1])])  for i in range(batch_stacks)]).to(torch.device("cpu:0"))
    else:
        random = torch.from_numpy(np.random.uniform(size = [Nmux*batch_stacks,1,shape_labeled[0]//n_repeat,shape_labeled[1]//n_repeat]))
        slm_stack = torch.repeat_interleave(random, n_repeat, dim = -1)
        slm_stack = torch.repeat_interleave(slm_stack, n_repeat, dim = -2)
        slm_stack = slm_stack.reshape(batch_stacks, Nmux, *shape_labeled )
    return slm_stack, random

def build_filteredweight_series(shape, sigma_list=[0.1, 0.2, 0.4, 0.7, 1, 2, 5], n_repeat=2): #sigma_list should have length = batch_stacks
    sigma_list = [s*n_repeat for s in sigma_list] # to roughly take into account the change of scale with n_repeat
    m = np.random.uniform(size=[shape[0]//n_repeat, shape[1]//n_repeat])
    m = np.repeat(np.repeat(m, n_repeat, axis=0), n_repeat, axis=1)
    mf = [m]
    for s in sigma_list:
        mf.append(gaussian_filter(m,sigma=s))
    mf.reverse()
    return mf

def arange_mask_series(mask_series):
    '''
    Input:  shape_labeled: [lx,ly] vector
            n_repeat: nb of px for the same value
    Output: slm_stack with the same random mask of shape [batch_stacks, Nx, Ny, Lx, Ly]     
    '''
    slm_stack = []
    for m in mask_series:
        slm_stack.append(torch.stack([ torch.from_numpy(m) for i in range(batch_layout[0]*batch_layout[1])]))
    slm_stack=torch.stack(slm_stack).to(torch.device("cpu:0"))
    return slm_stack   

def reshape_stack(stack):
    out=[]
    for i in range(batch_stacks):
        out.append(np.stack([stack[i*Nmux+j,0,::] for j in range(Nmux)]))
    return np.stack(out).swapaxes(0,1)

def bgr8_to_jpeg(frame, quality=75):
    return bytes(cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])[1])

def update_widget(img_widget, value):
    img_widget.value = bgr8_to_jpeg(value)

def exp( exp = 5000, widget = False, cal_key='uD-out#0' ):
    img_widget = ipywidgets.Image(format='jpeg', value=bgr8_to_jpeg(np.zeros((1,1))), width=400, height=300)

    #Initializes the SLM
    uD = DisplayGL(1, reverseX=False, reverseY=False)
    slm = DisplayGL(2, reverseX=False, reverseY=False)
    
    # Initialize and setup camera
    cam_dev = PylonCamera('40308273')
    cam_dev.configure(exposure=exp, trigLine="Line2", triggerDelay=2000, frameRate=120, reverseX=True, reverseY=False)# very high exposure for trigger
    dev_mgmt = DeviceManager(screens=[uD, slm], cameras=[cam_dev])
    cam_dev._camera.ExposureTime.Value = exp
    
    # Get the data from calibration
    os.chdir(calibdir)
    file = open(f'calibration_{calibfile}', 'rb')
    [cal_dict, slm_xy_offsets] = pickle.load(file) ; file.close()
    file = open(f'data_{calibfile}', 'rb')
    [zoom, img_size, slm_size, pat_pad, cam_pad, batch_layout, batch_stacks, shape_labeled, Npix] = pickle.load(file)
    file.close(); os.chdir(thisfiledir)

    if widget:
        train_wid_all = ipywidgets.Image(format='jpeg', value=bgr8_to_jpeg(np.zeros((1,1))), width=1000, height=550)
        train_wid_one = ipywidgets.Image(format='jpeg', value=bgr8_to_jpeg(np.zeros((1,1))), width=1000, height=550)
        CAM_layer = Camera(img_shape=(*batch_layout, *shape_labeled), batch_stacks=batch_stacks, device=device, cam_dev=cam_dev, cal_dict=cal_dict, cal_key=cal_key, update_widget = [lambda x: update_widget(train_wid_all, x), lambda x: update_widget(train_wid_one, x)])
    else:
        train_wid_all, train_wid_one = None, None
        CAM_layer = Camera(img_shape=(*batch_layout, *shape_labeled), batch_stacks=batch_stacks, device=device, cam_dev=cam_dev, cal_key=cal_key, cal_dict=cal_dict, update_widget=None)
    
    uD_layer = Screen(img_shape=(*batch_layout, *shape_labeled), batch_stacks=batch_stacks, xy_offsets=None, screen_dev=uD, cal_dict=cal_dict, device=device)
    SLM_layer = Mask(img_shape=(*batch_layout, *shape_labeled), batch_stacks=batch_stacks, xy_offsets=slm_xy_offsets, screen_dev=slm, cal_dict=cal_dict, device=device)
    
    class layer_model(object):

        def __init__(self):
            self.cam_dev = cam_dev
            self.SLM_layer = SLM_layer
            self.uD_layer = uD_layer
            self._dev_mgmt = dev_mgmt
            self.CAM_layer = CAM_layer
            self.zoom, self.Npix = zoom, Npix
            self.img_size, self.slm_size = img_size, slm_size
            self.cam_pad, self.pat_pad = cam_pad, pat_pad 
            self.batch_layout, self.batch_stacks = batch_layout, batch_stacks
            self.shape_labeled = shape_labeled
            self.Nmux= batch_layout[0]*batch_layout[1]
            
        def forward(self, imgs_uD, imgs_SLM):
            "Tensors with input format [batch_stacks, Nmux, *shape_labeled]"
            self.SLM_layer._weight.data = imgs_SLM
            self.uD_layer.forward(imgs_uD.reshape(self.Nmux*self.batch_stacks,1,*shape_labeled)) # different data format : why ? ask tim 
            self.SLM_layer.forward(None)
            self._dev_mgmt.start_capture(n_frame=self.batch_stacks, sync_frame=True, remove_ref=True)
            out = self.CAM_layer.forward(None)
            #self.out=out
            #out = torch.stack([torch.stack([out[imux+self.Nmux*i,0,::] for imux in range(self.Nmux)]) for i in range(self.batch_stacks)])
            return out #out.reshape(batch_stacks, *batch_layout, *Npix) #uncomment for out shape = (batch_stacks, *batch_layout, Npix_x, Npix_y)
        
    return layer_model(), [train_wid_all, train_wid_one]

os.chdir(thisfiledir)
