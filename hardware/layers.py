import cv2
import torch
import numpy as np
from scipy import signal
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


class FFLayer(ABC, nn.Module):
    """Abstract class defining typical Forward Forward mechanism such as the training function and the layerwise optimizer.
    """
    def __init__(self, optimizer, activation, device=torch.device("cpu"), requires_grad=True, FF_thres=4, lossfunc="FFclassic"):
        """
        :param optimizer: Lambda function to be use for the training of the layer (typically Adam)
        :param device: Device used for computing ('cpu' or 'cuda')
        :param requires_grad: If True, the layer weights will be updated. Otherwise the layer is fixed and can be used for example as a fixed convolution layer (such as a lens).        

        :return: Abstract FFLayer object
        """
        self._threshold = FF_thres
        self._activation = activation

        for param in self.parameters():
            param.requires_grad = requires_grad  
        
        self._optimizer = optimizer(self.parameters())
        self._lossfunc = lossfunc
        self._cosim = nn.CosineSimilarity(dim=1, eps=1e-6)

        p_vector = torch.normal(0, 1, size=(1, self.out_features))
        self._p_vector = p_vector / (p_vector.norm(2, 1, keepdim=True) + 1e-4)
        self._p_vector = self._p_vector.to(device)
        self._p_vector = (self._p_vector-torch.min(self._p_vector))/(torch.max(self._p_vector)-torch.min(self._p_vector))
        self._device = device
        self.to(device)

    def trainn(self, x_pos, x_neg, y, epoch, batch):
        """Training function for every Forward Forward based layers. It is the core mechanism of this model.
        
        The layer is fed with a positive and a negative data, the infered results is fed to a custom cost function (similar to SoftPlus). If the layer weights are trainable one step of the gradient is done.

        $\mathcal{L} = log\left(1 + e^{-\sum_j x_{{pos}_j}^2 + \sum_j x_{{neg}_j}^2}\\right)$
    
        :param x_pos: The positive data, input x with the matching label
        :param x_neg: The **negative** data, input x with the **incorrect** label

        :return: Forward pass of the positive and negative data to be fed to the next layer in the network.
        :rtype: torch.Tensor, torch.Tensor
        """
        g_pos = self.goodness(x_pos)
        g_neg = self.goodness(x_neg)
        y = y.to(self._device)
        
        if (self._lossfunc == "XEntropy"):
            self._loss = nn.CrossEntropyLoss()(x_pos, y)
            
        elif (self._lossfunc == "FFclassic"):
            pos_loss = -g_pos + self._threshold
            neg_loss = g_neg - self._threshold
            self._loss = torch.log(1 + torch.exp(torch.cat([pos_loss, neg_loss]))).mean()
            
        elif (self._lossfunc == "FFsymba"):
            self._loss = (torch.log(1 + torch.exp(-self._threshold*(g_pos - g_neg)))).mean()
            
        elif (self._lossfunc == "FFcossim"):
            self._loss = (torch.log(1 + torch.exp(-self._threshold*(g_pos - g_neg)))).mean()

        elif (self._lossfunc == "FFcossim_naive"):
            g_pos = nn.Flatten()(x_pos)
            g_neg = nn.Flatten()(x_neg)
            self._loss = torch.log(1+torch.exp(-self._threshold*(self._cosim(g_pos, g_neg)))).mean()
        
        # If not all parameters are not gradfree
        self._optimizer.zero_grad()
        self._loss.backward()        
        self._optimizer.step()

    @abstractmethod
    def forward(self, x):
        """

        :param x: nn.Tensor input tensor to apply the forward pass of the instanciated FFLayer

        :return: Forward pass of the instanciated FFLayer
        :rtype: nn.Tensor
        """
        return self._activation(x)

    def goodness(self, x):
        
        if(self._lossfunc in ["FFclassic", "FFsymba", "FFcossim_naive"]):
            return x.pow(2).view(x.shape[0], -1).mean(1)
            
        elif(self._lossfunc in ["FFcossim"]):
            return self._cosim(x, self._p_vector)


class FFResLayer(nn.Module):
    """Residual Layer for the Forward Forward method
    """
    
    def __init__(self, res_shape=(28, 28)):
        """
        :param layers: The list of layers to skip as a block
        :rtype: nn.Module
        """
        super().__init__()
        self._rshape = res_shape

    def forward(self, x, res):
        """ The input x is saved and summed with the layers to be skipped and summed at the end

        :param x: Input data to be processed and skip in regards with the layers
        :type x: torch.Tensor

        :return: 
        :rtype: torch.Tensor
        """
        out = x

        slice_x = slice((x.shape[-2]-self._rshape[-2])//2, (x.shape[-2]+self._rshape[-2])//2, 1)
        slice_y = slice((x.shape[-1]-self._rshape[-1])//2, (x.shape[-1]+self._rshape[-1])//2, 1)
        out[:,:,slice_x,slice_y] = res[:,:,:self._rshape[-2],:self._rshape[-1]]
        return out
        

class FFConv(FFLayer, nn.Conv2d):
    """2D convolution layer class object using the Forward Forward method
    In the simplest case, the output value of the layer with input size $(N,Cin,H,W)$ output $(N,Cout,Hout,Wout)$
    """
        
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, optimizer, activation, lossfunc="FFclassic",
                 bias=False, device=None, requires_grad=True, dtype=torch.float):
        """Applies a 2D convolution over an input signal composed of several input planes.

        :param in_channels: Number of channels in the input image
        :param out_channels: Number of channels produced by the convolution
        :param kernel_size: Size of the convolving kernel
        :param stride: Stride of the convolution. Default; 1
        :param padding: Padding added to all four sides of the input. Default; 0
        :param optimizer: Lambda function to be use for the training of the layer (typically Adam)
        :param bias: If True, adds a learnable bias to the output. Default; True
        :param device: Device used for computing ('cpu' or 'cuda')
        :param requires_grad: If True, the layer weights will be updated. Otherwise the layer is fixed and can be used for example as a fixed convolution layer (such as a lens).

        :return: Forward Forward 2D convolution layer object
        :rtype: nn.Module
        """
        nn.Conv2d.__init__(self,
                           in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=padding,
                           bias=bias,
                           device=device,
                           dtype=dtype)

        FFLayer.__init__(self, optimizer=optimizer, activation=activation, device=device, requires_grad=requires_grad, lossfunc=lossfunc)

    def forward(self, x):
        """Apply the usual 2D convolution operation on a given input x with the kernel weights of the current layer. The input must be normalized so that we don't biais the following layers.
        
        :param x: Input data to be processed
        :type x: torch.Tensor

        :return: 2D convolution of the input x with layer weights (kernel of the convolution)
        :rtype: torch.Tensor
        """

        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return super().forward(F.conv2d(x_direction, self.weight, bias=self.bias, stride=self.stride, padding=self.padding))


class FFLinear(FFLayer, nn.Linear):
    """Fully connected layer class object using the Forward Forward method
    """
    
    def __init__(self, in_features, out_features, optimizer, activation, bias=False, device=torch.device("cpu"), requires_grad=True, dtype=torch.float, lossfunc="FFclassic", FF_thres=3):
        """
        :param in_features: Size of each input sample
        :param out_features: Size of each output sample
        :param optimizer: Lambda function to be use for the training of the layer (typically Adam)
        :param bias: If True, adds a learnable bias to the output. Default; True
        :param device: Device used for computing ('cpu' or 'cuda')
        :param requires_grad: If True, the layer weights will be updated.
        :return: Forward Forward fully connected linear layer object
        :rtype: nn.Module
        """
        
        nn.Linear.__init__(self, in_features=in_features, out_features=out_features, bias=bias, device=device, dtype=dtype)
        FFLayer.__init__(self, optimizer=optimizer, activation=activation, device=device, requires_grad=requires_grad, lossfunc=lossfunc, FF_thres=FF_thres)

    def forward(self, x):
        """Applies a linear transformation to the inconming data: $y = x W^T + b$

        The input x is first normalized so that we don't biais the following layers and we focus on the input vector direction only.

        :param x: Input data to be processed
        :type x: torch.Tensor

        :return: Linear transformation of the input normalized x by the weight matrix 
        :rtype: torch.Tensor
        """
        
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        # return torch.matmul(x_direction, self.weight.T)
        return super().forward(nn.functional.linear(x_direction, self.weight, self.bias))
        

# https://stackoverflow.com/questions/51980654/pytorch-element-wise-filter-layer
class FFElmwise(FFLayer, nn.Linear):
    """Forward Forward element wise linear transformation layer.
    """
    
    def __init__(self, in_features, out_features, optimizer, activation, bias=False, device=None, requires_grad=True, dtype=torch.float, lossfunc="FFclassic"):
        """
        :param in_features: Size of each input sample
        :param out_features: Size of each output sample
        :param optimizer: Lambda function to be use for the training of the layer (typically Adam)
        :param bias: If True, adds a learnable bias to the output. Default; True
        :param device: Device used for computing ('cpu' or 'cuda')
        :param requires_grad: If True, the layer weights will be updated.
        :return: Forward Forward element wise linear layer object
        :rtype: nn.Module
        """
        nn.Linear.__init__(self, in_features=in_features, out_features=out_features, bias=bias, device=device, dtype=dtype)
        FFLayer.__init__(self, optimizer=optimizer, activation=activation, device=device, requires_grad=requires_grad, lossfunc=lossfunc)
        
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features).uniform_(to=1))

    def forward(self, x):
        """Applies a linear transformation to the inconming data: $y = x \odot W + b$

        Where $\odot$ is the element wise product

        The input x is first normalized so that we don't biais the following layers and we focus on the input vector direction only.

        :param x: Input data to be processed
        :type x: torch.Tensor

        :return: Linear transformation of the input normalized x by the weight matrix 
        :rtype: torch.Tensor
        """
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return super().forward(torch.mul(x_direction, self.weight))


class FFPhaseElmwise(FFLayer, nn.Linear):
    """Forward Forward implementation of the element wise phase linear transformation
    """
    
    def __init__(self, in_features, out_features, optimizer, activation, bias=False, device=None, requires_grad=True, dtype=torch.float, lossfunc="FFclassic"):
        """
        :param in_features: Size of each input sample
        :param out_features: Size of each output sample
        :param optimizer: Lambda function to be use for the training of the layer (typically Adam)
        :param bias: If True, adds a learnable bias to the output. Default; True
        :param device: Device used for computing ('cpu' or 'cuda')
        :param requires_grad: If True, the layer weights will be updated.
        :return: Forward Forward element wise linear layer object
        :rtype: nn.Module
        """
        nn.Linear.__init__(self, in_features=in_features, out_features=out_features, bias=bias, device=device, dtype=dtype)
        FFLayer.__init__(self, optimizer=optimizer, activation=activation, device=device, requires_grad=requires_grad, lossfunc=lossfunc)
        
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features).uniform_(to=2*np.pi))

    def forward(self, x):
        """Applies a linear transformation to the inconming data: $y = x \odot e^{j W} + b$

        Where $\odot$ is the element wise product

        The input x is first normalized so that we don't biais the following layers and we focus on the input vector direction only.

        :param x: Input data to be processed
        :type x: torch.Tensor

        :return: Linear transformation of the input normalized x by the weight matrix 
        :rtype: torch.Tensor
        """
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return super().forward(torch.mul(x_direction, torch.exp(1j*self.weight)))


class Diffuser(nn.Linear):
    """Fixed random phase diffuser with a circular aperture."""
    
    def __init__(self, in_features, out_features, device=torch.device("cpu"), D_smoothing=40, D_aperture=100):
        """
        :param in_features: Size of each input sample
        :param out_features: Size of each output sample
        :param D_smoothing: Diameter of the circular kernel used to correlate phase noise.
        :param D_aperture: Diameter of the circular Fourier-plane aperture.
        :return: Fixed diffuser module.
        :rtype: nn.Module
        """
        nn.Linear.__init__(self, in_features=in_features, out_features=out_features, bias=False, device=device)

        self._device = device
        self.to(self._device)
        
        # Define the smoothing kernel.
        self._kernel = np.zeros((D_smoothing, D_smoothing))
        self._aperture = np.zeros((in_features, out_features))
        
        self._kernel = cv2.circle(self._kernel, (D_smoothing//2, D_smoothing//2), D_smoothing//2, 1, -1)
        self._aperture = cv2.circle(self._aperture, (in_features//2, out_features//2), D_aperture//2, 1, -1)

        self._aperture = torch.from_numpy(self._aperture).to(self._device)
        
        # Perform a normalized convolution
        S = 2
        uncorrelated_phase = 2*S*np.pi*np.random.uniform(low = -1,high = 1, size = (in_features, out_features))
        correlated_phase = signal.fftconvolve(uncorrelated_phase, self._kernel, mode = 'same')/np.sqrt(np.pi*(D_smoothing/2)**2)
        
        self.weight.data =  torch.from_numpy(np.exp(1j*correlated_phase)).to(self._device)
        self.weight.requires_grad = False

    def forward(self, x):
        """Apply the diffuser phase mask and aperture to the input field.

        :param x: Input data to be processed
        :type x: torch.Tensor

        :return: Fourier-plane field after phase diffusion and aperture clipping.
        :rtype: torch.Tensor
        """
        
        # Pass the field incident on the focal plane through the aperture.
        incidentfield1 = torch.fft.fftshift(torch.fft.fft2(torch.mul(x, self.weight)))
        transmittedfield1 = torch.mul(incidentfield1, self._aperture)
        
        # Fourier transform the field transmitted through the focal plane to reach
        # the image plane.
        return transmittedfield1


class OpticProp(nn.Module):
    """Physical optical propagation layer combining screen, SLM mask, and camera readout."""
    
    def __init__(self, dev_mgmt, cal_dict, slm_xy_offsets, img_shape=(1, 1, 28, 28), batch_stacks=1, dtype=torch.float, device='cpu', update_widget=None):
        super().__init__()
        
        self._dev_mgmt = dev_mgmt
        self._cal_dict = cal_dict
        self._batch_stacks = batch_stacks

        self._screen = Screen(img_shape=img_shape, batch_stacks=batch_stacks, xy_offsets=None, screen_dev=dev_mgmt.get_screens()[0], cal_dict=cal_dict, device=device)
        self._mask = Mask(img_shape=img_shape, batch_stacks=batch_stacks, xy_offsets=slm_xy_offsets, screen_dev=dev_mgmt.get_screens()[1], cal_dict=cal_dict, device=device, sync_frame=True)
        self._cameras = [Camera(img_shape=img_shape,
                                batch_stacks=batch_stacks,
                                device=device,
                                cam_dev=camera,
                                cal_dict=cal_dict,
                                cal_key='uD-out#'+str(cam_id))
                         for cam_id, camera in enumerate(dev_mgmt.get_cameras())]
        
        self._cameras[0]._update_widget = update_widget

    def forward(self, x):
        self._screen.forward(x)
        self._mask.forward(None)

        # Get the timestamp value since the camera is already recording the Screen
        # There is a delay of 4 frames to consider, at 60 Hz -> 64 ms
        self._dev_mgmt.start_capture(n_frame=self._batch_stacks, sync_frame=True, remove_ref=True)
        img_out = []
        for camera in self._cameras:
            img_out.append(camera.forward(None))

        # Return the first camera observation
        return img_out[0]

        
class Camera(nn.Module):
    """Camera readout layer for physical acquisition or simulated intensity detection."""
    
    def __init__(self, img_shape=(1, 1, 28, 28), batch_stacks=1, dtype=torch.float, device='cpu', cam_dev=None, cal_dict=None, cal_key='uD-out#0', update_widget=None, remove_ref = True):
        super().__init__()

        (self._nh, self._nw, self._h, self._w) = img_shape
        self._batch_stacks = batch_stacks
        self._dtype = dtype
        self._device = device
        self._cam_dev = cam_dev
        self._cal_dict = {} if cal_dict is None else cal_dict
        self._cal_key = cal_key
        self._update_widget = update_widget
        self.remove_ref = remove_ref

        if(self._cam_dev):
            self._h, self._w = self._cal_dict['info'][cal_key + '_size']
            self._img_out = np.zeros((self._batch_stacks, len(self._cal_dict['data']), 1, self._h, self._w))
            self._img_out_cuda = torch.zeros((self._batch_stacks, len(self._cal_dict['data']), 1, self._h, self._w)).to(self._device).contiguous()

        # Ensure that the Camera layer is not trainable!
        for param in self.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        if(self._cam_dev is not None):
            # Storing the captures afterwards
            cam_read = self._cam_dev.get_img()
            for id_stack, img_cam in enumerate(cam_read):
                for item in self._cal_dict['data'].items():
                    
                    # Get the position data from calibration values
                    _, _, _, _, xmin, xmax, ymin, ymax = item[1][self._cal_key].values()
                    # Save using the corresponding calibration ID
                    self._img_out[id_stack, item[0], None, :, :] = img_cam[ymin:ymax, xmin:xmax]
    
                if(self._update_widget is not None):
                    self._update_widget[0](255*img_cam)
                    self._update_widget[1](255*img_cam[ymin:ymax, xmin:xmax])
                    
            # The final stacks of images is reshape to respect the initial batch_size (batch_size, x, y)
            self._img_out_cuda = torch.from_numpy(self._img_out).float().to(self._device).contiguous().reshape(self._batch_stacks * self._nh * self._nw, 1, self._h, self._w)
            return self._img_out_cuda

        else:
            # Apply the Camera intensity measure after this everything is done in electronic
            x = torch.abs(x)**2
            
            if(self._update_widget is not None):
                img_out = x.clone().cpu().detach().numpy()
                self._update_widget[0](img_out[0,0,:,:]*255)

            x = x.reshape(self._batch_stacks, self._nh, self._h, self._nw, self._w).swapaxes(2, 3)
            x = x.reshape(self._batch_stacks, self._nh * self._nw, self._h, self._w)
            return x.reshape(self._nh * self._nw * self._batch_stacks, 1, self._h, self._w)


class Screen(nn.Module):
    """Display layer that tiles input images onto a calibrated screen canvas."""
    
    def __init__(self, img_shape=(1, 1, 28, 28), batch_stacks=1, xy_offsets=None, dtype=torch.float, requires_grad=False, screen_dev=None, cal_dict=None, cal_entry='uD-in', device=torch.device('cpu'), sync_frame=True):
        super().__init__()

        (self._nh, self._nw, self._h, self._w) = img_shape
        self._batch_stacks = batch_stacks
        self._dtype = dtype
        self._device = device
        self._screen_dev = screen_dev
        self._cal_dict = cal_dict
        self._cal_entry = cal_entry
        self._xy_offsets = xy_offsets
        self._sync_frame = sync_frame

        self._h_padded, self._w_padded = self._h, self._w
        
        if(self._cal_dict is not None):
            self._h_padded = self._h + self._cal_dict['info']['padding'][0]
            self._w_padded = self._w + self._cal_dict['info']['padding'][1]

        self._tmp_padded = torch.zeros((self._nh*self._nw, self._batch_stacks, self._h_padded, self._w_padded), device=device).cpu()

        if(self._xy_offsets  is None):
            self._xy_offsets = [((self._w_padded - self._w)//2,
                                 (self._w_padded - self._w)//2,
                                 (self._h_padded - self._h)//2,
                                 (self._h_padded - self._h)//2) for _ in range(self._nh*self._nw)]

        if(self._screen_dev is not None):
            self._img_screen_tmp = torch.zeros((self._batch_stacks, *self._screen_dev.get_screen_size()), device=device).to(self._device).contiguous()
            
        for param in self.parameters():
            param.requires_grad = requires_grad
        
    def forward(self, x):
        x = x.to(torch.float64)
        # Reshape full batch of data into a stacks of imgs to be displayed continuously and captured then by the camera
        self._shape = x.shape
        if(x.device.type != 'cpu'):
            x = x.detach().cpu()

        x_stacks = x.reshape(self._batch_stacks, self._nh*self._nw, self._h, self._w).swapaxes(0,1)
        # Pad each tile according to the calibration offsets before composing the screen canvas.
        for k, _ in enumerate(x_stacks):
            self._tmp_padded[k] = torch.nn.functional.pad(x_stacks[k], self._xy_offsets[k], "constant", 0)

        # Reshaping the numpy array (self._nh*self._nw, self._batch_stacks, self._h, self._w)
        # To the shape (self._batch_stacks, *self._screen_dev.get_screen_size())
        tmp_img = self._tmp_padded.swapaxes(0,1)
        tmp_img = tmp_img.reshape(self._batch_stacks, self._nh, self._nw, self._h_padded, self._w_padded)
        tmp_img = tmp_img.swapaxes(2,3)
        tmp_img = tmp_img.reshape(self._batch_stacks, self._nh * self._h_padded, self._nw * self._w_padded)

        if(self._screen_dev is not None):
            # Suspend the screen to prepare the displaying
            self._screen_dev.set_render_ready(False)
        
            # Fit in the middle of the Screen Canvas
            half_h, half_w = np.shape(tmp_img)[1]//2, np.shape(tmp_img)[2]//2
            screen_h, screen_w = self._img_screen_tmp.size()[1]//2, self._img_screen_tmp.size()[2]//2
            self._img_screen_tmp[:, screen_h-half_h:screen_h+half_h, screen_w-half_w:screen_w+half_w] = tmp_img

            # Send the lists of images to be displayed
            self._screen_dev.display(self._img_screen_tmp, sync_frame=self._sync_frame, CUDA2GL=False)

            # Screen returns nothing if we use the optical setup
            return None
        
        else:
            return tmp_img.to(self._dtype).to(self._device)

class Mask(Screen):
    """SLM phase-mask layer displayed on a calibrated screen device."""
    def __init__(self, img_shape=(1, 1, 28, 28), batch_stacks=1, xy_offsets=None, dtype=torch.float, requires_grad=False, screen_dev=None, cal_dict=None, cal_entry='slm-in', device=torch.device('cpu'), sync_frame=True):
        
        super().__init__(img_shape=img_shape,
                         batch_stacks=batch_stacks,
                         xy_offsets=xy_offsets,
                         dtype=dtype,
                         requires_grad=requires_grad,
                         screen_dev=screen_dev,
                         cal_dict=cal_dict,
                         cal_entry=cal_entry,
                         device=device, sync_frame=sync_frame)
        
        self._weight = nn.Parameter(torch.Tensor(self._nh * self._nw * self._batch_stacks, self._h, self._w).uniform_(to=1))
        if(self._screen_dev is not None):
            #self._img_screen_tmp = torch.zeros((self._batch_stacks, *self._screen_dev.get_screen_size()), device=device).to(self._device).contiguous()
            im_zero = torch.zeros(self._screen_dev.get_screen_size(), device=device).to(self._device).contiguous()
            (sx, sy) = self._screen_dev.get_screen_size()
            XX, YY = np.meshgrid(np.arange(sx), np.arange(sx),indexing='ij')
            im_alt = ((XX%2).astype('bool')^(YY%2).astype('bool'))
            self.im_alt = im_alt
            im_zero[:,(sy-sx)//2:(sy-sx)//2+sx]= torch.from_numpy(im_alt).to(torch.float64)
            self._img_screen_tmp = torch.stack([im_zero for i in range(self._batch_stacks)]).to(self._device).contiguous()
            
    def forward(self, x):
        super().forward(self._weight)
