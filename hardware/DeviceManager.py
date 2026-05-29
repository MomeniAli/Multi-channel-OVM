import cv2
import time
import numpy as np
import subprocess


__queue_length__ = 32
__ref_frame_tol__ = 2
__xrandr_Ntry__ = 10
__screen_rst_time__ = 300
__default_camera_shape__ = (1200, 1920)


def _camera_frame_shape(camera):
    camera_obj = getattr(camera, "_camera", None)
    height_node = getattr(camera_obj, "Height", None)
    width_node = getattr(camera_obj, "Width", None)
    try:
        return int(height_node.Value), int(width_node.Value)
    except Exception:
        return __default_camera_shape__


class DeviceManager(object):
    """
        Interface for the screens and cameras manager
    """
    def __init__(self, screens=None, cameras=None):
        """
        Initialize the DeviceManager providing the screens and cameras to be interfaced
        
        :param screens: List of DisplayGL objects to be managed
        :param cameras: Listo f PylonCamera objects to be managed

        :return: Instance of a DeviceManager interface
        :rtype: DeviceManager
        """
        # Register the screens that the camera is master
        self._screens = list(screens or [])
        self._cameras = list(cameras or [])
        
        # For frame synchronization
        self._reset_time = time.time()       

        # Resynchronize the screens
        self.reset_screen_trigger(sync_time=True)
    
    def get_screens(self):
        """ Return the screen object of DeviceManager
        
        :return: self._screens
        :rtype: DisplayGL
        """
        return self._screens

    def get_cameras(self):
        """ Return the cameras object of DeviceManager
        
        :return: self._cameras
        :rtype: PylonCamera
        """
        return self._cameras
    
    def capture_ref(self):
        """ Makes a capture to store as the internal threshold value for the SYNC Frame value comparison
        
        :return: None
        """
        # Clear the system to be in a controlled state
        for screen in self._screens:
            screen.clear()

        for camera in self._cameras:
            camera.clear()
            time.sleep(2)
            camera.set_ref_img()

    def sync_time_calibration(self):

        # Clear the system to be in a controlled state
        for screen in self._screens:
            screen.clear()

        for camera in self._cameras:
            camera.clear()
        
        self._sync_slice = []
        for _ in range(__xrandr_Ntry__):
            # Display and let enough time for the camera to see the pattern
            img_cap_lst = []
            uD = self._screens[0]
            mCam = self._cameras[0]

            for id_mask, mask in enumerate(uD._sync_mask):
                uD.set_render_ready(True)
                uD.displayCUDA(imageData=mask * 255, sync_frame=False)
                mCam.clear()

                time.sleep(2)
                img_cap_lst.append(mCam.get_raw_img()[0])
            
            # Post-processing getting the rectangle contour and deduce the coordinates
            for img in img_cap_lst:
                # Apply Gaussian blur to reduce noise and improve edge detection
                blurred = cv2.GaussianBlur((img * 255).astype(np.uint8), (15, 15), 0)
                
                # Otsu's thresholding after Gaussian filtering
                ret, thresh = cv2.threshold(blurred, 50, 255, 0)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(51, 51))
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

                # Find contours in the edged image 
                contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
                 
                # Loop over the contours 
                for contour in contours:
                    M = cv2.moments(contour)
                    if(M["m00"] > 64**2):
                        # Approximate the contour to a polygon 
                        epsilon = 0.1 * cv2.arcLength(contour, True) 
                        approx = cv2.approxPolyDP(contour, epsilon, True) 
                         
                        # Check if the approximated contour has 4 points (rectangle) 
                        if len(approx) == 4: 
                            x, y, w, h = cv2.boundingRect(contour)
                            #print(x, y, w, h)
                            self._sync_slice.append({'x': slice(x, x+w), 'y': slice(y, y+h)})
        
            if(len(self._sync_slice) != 4):
                print("Couldn't find all calibration pattern for camera")
            else:
                break

        img_ref = self._cameras[0].get_ref_img()
        self._sync_threshold = np.zeros(len(self._sync_slice))
        for k in range(len(self._sync_slice)):
            self._sync_threshold[k] = np.sum(img_ref[self._sync_slice[k]['y'], self._sync_slice[k]['x']])*__ref_frame_tol__
    
    def start_capture(self, n_frame=1, sync_frame=False, remove_ref=False):
        """ Return the copy of the buffer from the threaded method: self._img_handler.OnImageGrabbed()
        
        :return: Copy of the acquired image
        :rtype: np.array([buffer.Height(),  buffer.Width()])
        """
        # Periodically reset the screens to keep the camera/display trigger in sync.
        if((time.time() - self._reset_time) >= __screen_rst_time__):
            self.reset_screen_trigger()
        
        n, n_try = 0, 0
        frame_h, frame_w = (
            _camera_frame_shape(self._cameras[0]) if self._cameras else __default_camera_shape__
        )
        img_out = np.zeros((len(self._cameras), n_frame, frame_h, frame_w))

        # Time control
        cam_trig_delta = np.zeros(n_frame)
        trig_time = np.zeros(len(self._cameras))
        cam_time_start = np.zeros(len(self._cameras))
        
        for cam_id, camera in enumerate(self._cameras):
            camera.clear()
            cam_time_start[cam_id] = camera.get_timestamp()

        # Start acquisition
        self.trigger_screens()
        
        while(n < n_frame):
        
            if(not sync_frame):
                for cam_id, camera in enumerate(self._cameras):
                    img_out[cam_id][n] = camera.get_raw_img(remove_ref)[0]
                n = n + 1

            else:              
                screen_reset = True
                # Browse through the data until we have a frame (skipping empty images)
                for _ in range(__queue_length__):

                    for cam_id, camera in enumerate(self._cameras):
                        img_out[cam_id][n], trig_time[cam_id] = camera.get_raw_img(remove_ref)

                    # Maybe make a tiny  function for the following line?
                    n_slice = n % len(self._sync_slice)
                    trig_sum = np.sum(img_out[0][n][self._sync_slice[n_slice]['y'], self._sync_slice[n_slice]['x']])
                    
                    if(trig_sum > self._sync_threshold[n_slice]):
    
                        # Check that the frame is coming from after 10ms after the trigger of the screens
                        delta_t_frame = trig_time[0] - cam_time_start[0]
                        if(delta_t_frame > 9e7):
                            screen_reset = False
                            cam_trig_delta[n] = delta_t_frame
                            break

                # Check that we dont have a duplicate!
                if((n_frame > 1) and (n+1 == n_frame) and (np.diff(cam_trig_delta/1e6).std() > 1)):
                    #print("Issue with synchronisation, re-acquiring")
                    screen_reset = True
                    
                if(screen_reset):
                    n = 0
                    n_try += 1
                    # If after __detect_sync_frame__ trials we didnot get anything we reset the screen
                    # We make a new capture which suspend the screen and reset its queue then we reload
                    # the Cuda Tensor onto the C++ DisplayGL wrapper to retry
                    if(n_try > __xrandr_Ntry__):
                        #print("Resetting screens...")
                        self.reset_screen_trigger()
                        n_try = 0
                        
                    # When everything is ready, we get a capture for the ref of the Camera
                    for cam_id, camera in enumerate(self._cameras):
                        camera.clear()
                        cam_time_start[cam_id] = camera.get_timestamp()
                    self.trigger_screens()
                else:
                    n = n + 1

        # Store the acquired images in their respective camera and pause the screens
        for cam_id, camera in enumerate(self._cameras):
            camera.store_img(img_out[cam_id])

        self.pause_screens()

    def trigger_screens(self):
        """ Trigger the display of the stack of images of all screens that the camera control
        
        :return: None
        """
        self._screens[0].displayCUDA()
        for screen in self._screens:
            screen.set_render_ready(True)

    def pause_screens(self):
        """ Pause the display to avoid displaying before the aquisition is ready
        
        :return: None
        """
        for screen in self._screens:
            screen.clear()
            screen.set_render_ready(False)

    def reset_screen_trigger(self, sync_time=False):
        """ Reset the trigger signals of all screens to be synchronized back to original state
        This should be unfortunately done every 10mins or so.
        
        :return: None
        """
        self.pause_screens()
        mScreen = self._screens[0]
        mScreen.pause_render()

        # Power-cycle the display outputs before restoring the configured layout.
        time.sleep(0.25)
        subprocess.run(["xrandr", "--output", "DP-1", "--off", "--nograb",
                                  "--output", "DP-2", "--off", "--nograb"])
        for n_try in range(__xrandr_Ntry__):
            time.sleep(1)
            sub_proc = subprocess.run(["xrandr", "--output", "DP-6", "--mode", "2560x1440",
                                       "--output", "DP-2", "--mode", "1920x1080",  "--right-of", "DP-6", "--nograb",
                                       "--output", "DP-1", "--mode", "1920x1200", "--right-of", "DP-2", "--nograb"])
            try:
                sub_proc.check_returncode()
            except subprocess.CalledProcessError:
                print("Error while restarting the screen")
                print("Attempt:", n_try)
                continue

            else:
                break

            finally:
                pass
        
        # When the screen is back, we can restore and render on the window on its corresponding screen as before
        time.sleep(1)
        mScreen.restore_window()
        time.sleep(0.1)
        mScreen.start_render()
        time.sleep(0.1)
        self.trigger_screens()

        # We also reset the counter for autoreset
        self._reset_time = time.time()
        
        # When everything is ready, we get a capture for the ref of the Camera
        self.capture_ref()
        if(sync_time):
            self.sync_time_calibration()
