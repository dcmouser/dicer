# !/usr/bin/python

# dicer - dicecamera.py
# version 1.0, 2/11/16
# mouser@donationcoder.com


# -----------------------------------------------------------
# dice camera helper class
# -----------------------------------------------------------

# for open cv
import cv2

# from us
import dicerfuncs

# generic
import time



class DiceCamera:
    """Class uses to manage camera."""

    def __init__(self):
        # derived class will implement
        pass

    def allowCameraToFocus(self,flag_isimportant):
        # derived class will handle
        pass

    def release(self):
        # derived class will handle
        pass

    def read(self):
        # derived class will handle
        return False, None


    @staticmethod
    def createCamera():
        # class function
        # create new DiceCamera, either opencv version or picamera version
        # try to create picamera, if we fail, fall back open opencv camera


        if (False):
            # bypass pi camera
            return DiceCamera_OpenCv()

        try:
            import picamera
        except:
            # exception importing picamera means we fall back to opencv camera
            return DiceCamera_OpenCv()

        # picamera imported, so we use picamera
        return DiceCamera_PiCamera()












class DiceCamera_OpenCv(DiceCamera):
    """Derived from DiceCamera."""

    def __init__(self):
        DiceCamera.__init__(self)
        self.capDevice = cv2.VideoCapture(0)


    def allowCameraToFocus(self, flag_isimportant):
        # derived class will handle
        """Give webcam time to settle and auto exposure, etc."""
        if (flag_isimportant):
            waittime = 30
        else:
            waittime = 5
        windowname = 'Warming up OpenCv camera...'
        # give camera time to settle
        for i in range(1, waittime):
            # Capture frame and ignore
            ret, frame = self.read()
            if (flag_isimportant):
                dicerfuncs.cvImgShow(windowname, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyWindow(windowname)


    def release(self):
        self.capDevice.release()

    def read(self):
        return self.capDevice.read()






class DiceCamera_PiCamera(DiceCamera):
    """Derived from DiceCamera.
    See http://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/"""

    def __init__(self):
        DiceCamera.__init__(self)
        import picamera
        from picamera.array import PiRGBArray

        # see http://picamera.readthedocs.org/en/release-1.10/fov.html

        # camera resolution
        self.camResolution = (640,640)
        #self.camResolution = (320,320)
        #self.camResolution = (2592,1944)

        self.camUseVideoMode = True

        # create picamera object (we get a deprecation warning if we dont set res this way)
        self.picamDevice = picamera.PiCamera(resolution=self.camResolution)

        # fix values so they dont auto adjust -- see http://picamera.readthedocs.org/en/release-1.10/recipes1.html#capturing-consistent-images
        self.picamDevice.framerate = 5

        # Wait for the automatic gain control to settle
        time.sleep(2)

        # Now fix the values
        self.picamDevice.shutter_speed = self.picamDevice.exposure_speed
        self.picamDevice.exposure_mode = 'off'
        g = self.picamDevice.awb_gains
        self.picamDevice.awb_mode = 'off'
        self.picamDevice.awb_gains = g

        # allocate array for reuse
        self.rawCapture = PiRGBArray(self.picamDevice, size=self.camResolution)

    def allowCameraToFocus(self, flag_isimportant):
        # derived class will handle
        """Give webcam time to settle and auto exposure, etc."""
        if (flag_isimportant):
            waittime = 10
        else:
            waittime = 2
        windowname = 'Warming up PiCamera...'
        #print windowname
        # give camera time to settle
        for i in range(1, waittime):
            # Capture frame and ignore
            #print "waiting %d" % i
            ret, frame = self.read()
            if (flag_isimportant):
                dicerfuncs.cvImgShow(windowname, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyWindow(windowname)


    def release(self):
        #self.picamDevice.stop_preview()
        self.picamDevice.close()

    def read(self):
        #import picamera
        #from picamera.array import PiRGBArray
        # do new need to allocate this each time!
        if (True):
            from picamera.array import PiRGBArray
            self.rawCapture = PiRGBArray(self.picamDevice, size=self.camResolution)
        # capture
        self.picamDevice.capture(self.rawCapture, format="bgr", use_video_port=self.camUseVideoMode)
        image = self.rawCapture.array
        #self.picamDevice.stop_preview()
        return (True, image)
