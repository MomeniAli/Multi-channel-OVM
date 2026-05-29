import os
import cv2
import time
import torch
import queue
import subprocess
from collections import deque
import numpy as np
from pypylon import pylon
from pypylon import genicam


__queue_length__ = 16
__cam_nbuffer__ = 8

class DeQueue:
    def __init__(self, maxsize): 
        self.deque = deque(maxlen=maxsize)
    
    def put(self, elem):
        self.deque.append(elem)
    
    def get(self):
        while(not self.deque):
            time.sleep(0.001)
        return self.deque.popleft()

    def clear(self):
        self.deque.clear()

class ImageEventPrinter(pylon.ImageEventHandler):

    def __init__(self, queue_length=0): # If queue_length=0 then it is an infinite queue
        super().__init__()
        self._converter = pylon.ImageFormatConverter()
        self._converter.OutputPixelFormat = pylon.PixelType_Mono8
        self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        self._imgQueue = DeQueue(maxsize=queue_length)

    def OnImagesSkipped(self, camera, countOfSkippedImages):
        print("OnImagesSkipped event for device ", camera.GetDeviceInfo().GetModelName())
        print(countOfSkippedImages, " images have been skipped.")

    def OnImageGrabbed(self, camera, grabResult):
        if grabResult.GrabSucceeded():
            self._imgQueue.put([grabResult.GetArray()/255, grabResult.ChunkTimestamp.Value])
            grabResult.Release()
        else:
            print("Error: ", grabResult.GetErrorCode(), grabResult.GetErrorDescription())

    def clear_FIFO(self):
        """
        Clear the Pylon Camera FIFO to remove unecessary old stuff
        
        :return: None
        """
        self._imgQueue.clear()


class PylonCamera(object):
    """
        Interface for the Pylon Basler Camera
    """
    def __init__(self, serial_id):

        # Look for camera devices
        self._tlf = pylon.TlFactory.GetInstance()
        dev_lst = self._tlf.EnumerateDevices()
        camID_lst = list(map(lambda x: x.GetSerialNumber(), dev_lst))
        print("Camera devices ID found:", camID_lst)
        
        # Declare new instance of Pylon camera
        self._camera = pylon.InstantCamera(self._tlf.CreateDevice(dev_lst[camID_lst.index(serial_id)]))
        self._camera.Open()

        # to get consistant results it is always good to start from "power-on" state
        self._camera.UserSetSelector.Value = "Default"
        self._camera.UserSetLoad.Execute()
        
        # Camera event processing must be activated first, the default is off.
        self._camera.MaxNumBuffer = __cam_nbuffer__
        self._img_handler = ImageEventPrinter(__queue_length__)
        self._camera.RegisterImageEventHandler(self._img_handler, pylon.RegistrationMode_Append, pylon.Cleanup_Delete)

        # For frame synchronization
        self._reset_time = time.time()

        #
        self._stored_imgs = None
    
    def __del__(self):
        """ Destructor to close the Basler camera object
        
        :return: None
        """
        # Releasing the resource
        self._camera.StopGrabbing()
        self._camera.DeregisterImageEventHandler(self._img_handler)
        self._camera.Close()

    def configure(self, exposure=5000, trigLine="Line2", triggerDelay=1500, frameRate=120, reverseX=True, reverseY=True):
        """ Function to open and connect to the camera, it automatically look for an available device and take the first one of the list

        :param exposure: Exposure value of the camera
        :param frameRate: FPS recording of the camera
        :param ReverseX: Boolean whether to flip horizontally
        :param ReverseY: Boolean whether to flip vertically
        
        :return: True if connection to the camera was sucessful otherwise False
        :rtype: Bool
        """
       
        # SPecify framerate to match display and avoid aliasing
        self._camera.AcquisitionFrameRate.Value = frameRate
        self._camera.AcquisitionFrameRateEnable.Value = True
        
        # Set the exposure time in ms
        self._camera.ExposureTime.Value = exposure
        self._camera.Gain.Value = 0.5

        self._camera.ReverseX.Value = reverseX
        self._camera.ReverseY.Value = reverseY

        self._camera.TriggerSelector.Value = "FrameStart"
        self._camera.TriggerMode.Value = "On"
        self._camera.TriggerSource.Value = trigLine
        self._camera.TriggerActivation.Value = "FallingEdge"
        self._camera.TriggerDelay.Value = triggerDelay

        try:
            self._camera.ChunkModeActive.Value = True
            self._camera.ChunkSelector.Value = "Timestamp"
            self._camera.ChunkEnable.Value = True
        except pylon.AccessException:
            pass

        # Grabing Continusely (video) with minimal delay
        try:
            self._camera.StartGrabbing(pylon.GrabStrategy_OneByOne, pylon.GrabLoop_ProvidedByInstantCamera)
        except:
            print('Camera already grabbing')
            pass

    def set_ref_img(self):
        """ Acquire a reference image for further substraction when acquiring images
        
        :return: None
        :rtype: None
        """
        self._ref_img = self.get_raw_img(remove_ref=False)[0]

    def get_ref_img(self):
        """ Return the reference image to obtain the threshold value required by DeviceManager
        
        :return: Copy of the reference image
        :rtype: np.array([buffer.Height(),  buffer.Width()])
        """
        return self._ref_img
    
    def get_raw_img(self, remove_ref=False):
        """ Return the copy of the buffer from the threaded method: self._img_handler.OnImageGrabbed()
        
        :return: Copy of the acquired image
        :rtype: np.array([buffer.Height(),  buffer.Width()])
        """
        if(remove_ref):
            img, timestamp = self._img_handler._imgQueue.get()
            return cv2.subtract(img, self.get_ref_img()), timestamp
        else:
            return self._img_handler._imgQueue.get() 

    def store_img(self, img):
        """ Store the images after the pre-processing done by DeviceManager
        
        :return:None
        """
        self._stored_imgs = img

    def get_img(self, remove_ref=False):
        """ Return the stored imaged after pre-processing done by DeviceManager
        
        :return: Acquired image and pre processed by DeviceManager
        :rtype: np.array([buffer.Height(),  buffer.Width()])
        """
        if(self._stored_imgs is None):
            raise Exception("There is no images stored in this camera") 

        return self._stored_imgs
            
    def clear(self):
        """ Clear the Pylon Camera FIFO to remove unecessary old stuff
        
        :return: None
        """
        self._img_handler.clear_FIFO()
        self._stored_imgs = None

    def get_timestamp(self):
        """ Execute and return the timestamp of the camera
        
        :return: TimeStamp of the camera
        :rtype: float
        """
        self._camera.TimestampLatch.Execute()
        return self._camera.TimestampLatchValue.Value