# !/usr/bin/python

# dicer - diceprofile.py
# version 1.0, 1/11/16
# mouser@donationcoder.com


# -----------------------------------------------------------
# dice profile class
# -----------------------------------------------------------

# imports
import os
import re
import time
import math
# us
import dicerfuncs
import dicecamera
import diecontrol
import dieextract
# for open cv
import numpy as np
import cv2



# -----------------------------------------------------------
# HARDWARE CONFIG
#global global_ArduinoComPort
global_ArduinoComPort = 'COM6'
# -----------------------------------------------------------









class DieProfile:
    """Class uses to hold info about a die being analzed."""

    def __init__(self):
        # what shape is it, and how many sides does it have?
        self.shape = None
        # how many unique faces
        self.dieFaces = None
        # is the face made of labels (lines) or full graphics (non-lines)
        self.graphicType = None
        # color information?
        self.colorBackground = None
        self.colorForeground = None
        # unique label for subdirectory naming for storing images, etc.
        self.uniqueLabel = None
        # for quantizing colors to extract foreground
        self.quantizedColorClusters = None
        self.quantizedClosestColorIndex = None
        self.quantizedColorSpace = None
        # arbitrary parameter hints
        self.parameterHints = list()
        # list of image files (cached)
        self.fileList = {}
        #
        self.colorPick = False
        #
        self.maxDistanceContourAdd = None
        self.maxDistanceFaceCentroidAdd = None
        self.maxDistanceFaceCentroidAddFar = None
        #
        # helper date for autosave
        self.autoSaveLastSeenTime = None
        self.autoSaveLastImg = None
        self.autoSaveLastSeenImg = None
        #
        # setting these to True bypasses our normal attempt to extract foreground labels
        # use full image
        self.useFullDieImage = False
        # dont threshold foreground
        self.dontThresholdForeground = False
        #
        self.diecenter = False
        #
        self.cropWidth = None
        self.cropHeight = None
        self.cropCircleX = 0.0
        self.cropCircleY = 0.0
        self.cropCircleR = 0.0
        #
        self.diceCamera = None
        #
        self.autoRoll = None
        self.autoRollHelper = None
        #
        self.denoise = None
        #
        self.backgroundDirectory = dicerfuncs.get_defaultBackgroundDirectory()

    def set_uniqueLabel(self, val):
        self.uniqueLabel = val

    def set_dieShape(self, val):
        self.shape = val

    def set_colorPick(self, val):
        self.colorPick = val

    def set_dieFaces(self, val):
        self.dieFaces = val

    def set_colorForeground(self, val):
        self.colorForeground = val

    def set_cropWidth(self, val):
        self.cropWidth = val
    def set_cropHeight(self, val):
        self.cropHeight = val

    def set_cropCircle(self, x,y,r):
        self.cropCircleX = x
        self.cropCircleY = y
        self.cropCircleR = r

    def set_quantizedColors(self, colorClusters, closestColorIndex, colorSpace):
        self.quantizedColorClusters = colorClusters
        self.quantizedClosestColorIndex = closestColorIndex
        self.quantizedColorSpace = colorSpace


    def set_dieFull(self, val):
        """This controls wheteher we use full image of die or just isolate foreground."""
        self.useFullDieImage = val

    def set_diecenter(self, val):
        self.diecenter = val
    def get_diecenter(self):
        return self.diecenter

    def set_dieThreshold(self, val):
        self.dontThresholdForeground = val


    def set_autoRoll(self, val):
        """For autorolling helper."""
        self.autoRoll = val

    def get_autoRoll(self):
        """For autorolling helper."""
        return self.autoRoll


    def set_denoise(self, val):
        self.denoise = val
    def get_denoise(self):
        return self.denoise


    def update_autoSaveLastImg(self, val):
        self.autoSaveLastImg = val

    # self.autoSaveLastTime = time.time()
    def update_autoSaveLastSeenImg(self, val):
        self.autoSaveLastSeenImg = val
        self.autoSaveLastSeenTime = time.time()

    def get_uniqueLabel(self):
        return self.uniqueLabel

    def get_subdir(self, subdir):
        return self.get_basefdir() + '/' + subdir


    def get_fileList(self, flag_useCache, subdir, repattern):
        if (not flag_useCache) or (not subdir in self.fileList):
            basefdir = self.get_subdir(subdir)
            #print "Using basedir: %s" % basefdir
            self.fileList[subdir] = self.buildRecursiveImageFileListFromDirectoryPattern(basefdir, repattern)
        return self.fileList[subdir]

    def get_dieFaces(self):
        return self.dieFaces

    def get_colorForeground(self):
        return self.colorForeground

    def get_basefdir(self):
        return self.get_diceDirectory() + self.uniqueLabel

    def get_diceDirectory(self):
        """Return base output directory, terminated with dir separator."""
        return self.get_outputDirectory() + "dice/"

    def get_colorPick(self):
        return self.option_colorPick

    def get_quantizedColors(self):
        return (self.quantizedColorClusters, quantizedClosestColorIndex)

    # def get_autoSaveLastTime(self):
    #	return self.autoSaveLastTime
    def get_autoSaveLastImg(self):
        return self.autoSaveLastImg

    def get_autoSaveLastSeenImg(self):
        return self.autoSaveLastSeenImg

    def calc_autoSaveElapsedTimeSinceLastSeen(self):
        if (self.autoSaveLastSeenTime is None):
            return 0.0
        else:
            return time.time() - self.autoSaveLastSeenTime

    def get_outputDirectory(self):
        """Return base output directory, terminated with dir separator."""
        return ""

    def get_faceImageDir(self, imageid):
        # get label from die
        basefdir = self.get_basefdir()
        return basefdir + "/" + str(imageid)

    def get_maxDistanceContourAdd(self):
        return self.maxDistanceContourAdd
    def get_maxDistanceContourAddFar(self):
        return self.maxDistanceContourAddFar

    def get_maxDistanceFaceCentroidAdd(self):
        return self.maxDistanceFaceCentroidAdd

    def get_cropWidthHeight(self):
        return (self.cropWidth, self.cropHeight)



    def get_useFullDieImage(self):
        return self.useFullDieImage
    def get_dontThresholdForeground(self):
        return self.dontThresholdForeground






    # -----------------------------------------------------------
    def get_backgroundFilepath(self):
        """Get path of background filename."""
        backgroundFilepath = self.get_backgroundDirectory() + "/" + dicerfuncs.get_backgroundFilename()
        return backgroundFilepath

    def get_backgroundDirectory(self):
        return self.backgroundDirectory

    def set_backgroundDirectory(self, val):
        self.backgroundDirectory = val

    def useBackgroundFileIfPresentInSubdir(self, imagesubdir):
        """If a background image is found in this subdirectory, use IT instead of the default background image."""
        bdir = self.get_basefdir() + "/" + imagesubdir
        # set it (whether exists or not)
        self.backgroundDirectory = bdir
        print "Using background image in %s" % bdir

    def get_reportDirectory(self):
        """Return the directory where reports should be written."""
        return self.get_basefdir()
    # -----------------------------------------------------------







    # helper funcs

    def buildRecursiveImageFileListFromDirectoryPattern(self, rootdir, repattern):
        """Give directory path (that might have wildcard) recursively find a list of all images, return as list."""
        prog = re.compile(repattern)
        filelist = list()
        for root, dirs, files in os.walk(rootdir):
            for name in files:
                result = prog.match(name)
                if not (result is None):
                    fpath = os.path.join(root, name)
                    filelist.append(fpath)
        return filelist



    def makeUniqueFilename(self, basefpath, fname):
        self.myMakeDirPath(basefpath)
        fpath = self.myMakeUniqueFilename(basefpath, fname)
        return fpath

    def myMakeDirPath(self, fpath):
        """Make a directory path."""
        try:
            os.makedirs(fpath)
        except OSError as exc:  # Python >2.5
            pass
        # if exc.errno == errno.EEXIST and os.path.isdir(path):
        #	pass
        # else:
        #	raise
        except:
            raise

    def myMakeUniqueFilename(self, fdir, fname):
        """Get unique filename.
        See http://code.activestate.com/recipes/577200-make-unique-file-name/
        """
        maxnum = 9999
        name, ext = os.path.splitext(fname)
        make_fn = lambda i: os.path.join(fdir, '%s_%04d%s' % (name, i, ext))
        for i in xrange(1, maxnum):
            uni_fn = make_fn(i)
            if not os.path.exists(uni_fn):
                return uni_fn
        return None

    def calc_dieFaceIdFromPath(self, fpath):
        """Given a full file path, return the guessed die face id."""
        # we want the last subdirectory before image filename so basically regex to grab last other\THIS\other
        searchObj = re.search(r'[^\\\/]+[\\\/]([^\\\/]+)[\\\/][^\\\/]+$', fpath, re.M | re.I)
        if (searchObj):
            faceId = searchObj.group(1)
        else:
            faceId = "unknown"
        return faceId

    def changeExtension(self, fpath, newExtension):
        """Remove .whatever and repalce with newExtension."""
        pattern = r'([^\.]+)(\.[^\.]+)$'
        replace = r'\1' + newExtension
        fpath = re.sub(pattern, replace, fpath)
        return fpath

    def computeFaceCenter(self, hull):
        centroid = dicerfuncs.findHullMomentCentroid(hull)
        return centroid

    def makeForegroundExtractionMask(self, img, mask, hull):
        """Given a starting mask and hull, make a mask that helps us decide which areas COULD hold foreground info.
        If entire area then just return mask
        """

        # no processing?
        # ATTN: in future we might want to so some minimal img cropping
        if (self.get_useFullDieImage()):
            # just return img and mask
            return (img, mask)


        mask = dicerfuncs.copyCvImage(mask)
        centroid = self.computeFaceCenter(hull)

        (height, width) = mask.shape[:2]
        maxside = max(height, width)

        # starting and mask
        #imgAnd = dicerfuncs.makeBinaryImageMaskForImg(mask)

        # the mask we make may be dependent on self.shape
        if (self.shape is None) or (self.shape == "circle"):
            # circular shape
            radiusAll = min(centroid[0], centroid[1])
            # ATTN: 2/24/16 this possibly should be a bit smaller circle like / 1.6, but tht can mess with some 2-digit extractions
            #radius = int(radiusAll / 1.5)
            # ATTN: 2/25/16 1.5 worked on our old die, 1.4 needed on new one
            radius = int(radiusAll / 1.4)

            # mask it
            (img, mask) = self.applyForegroundExtractionMask_Circle(img,mask,centroid,radius)
            #color = 255
            #cv2.circle(imgAnd, centroid, radius, color, thickness=-1)
            #mask = cv2.bitwise_and(imgAnd, mask)

            # other parameters we can be queried
            # was 16 as of 2/5/16 but this was rejected periods near 9s
            # self.maxDistanceContourAdd = maxside / 1.0


            # 2/24/16:
            #self.maxDistanceContourAdd = maxside / 12
            self.maxDistanceContourAdd = maxside / 12
            # 2/25/16 had to change this from 5 to 4 for new die
            self.maxDistanceContourAddFar = maxside / 5

            # was 52 as of 2/24/16
            #self.maxDistanceFaceCentroidAdd = maxside / 52
            # ATTN: 2/25/16 -- needed for new die
            #self.maxDistanceFaceCentroidAdd = maxside / 12
            self.maxDistanceFaceCentroidAdd = maxside / 18


        elif (self.shape == "square"):
            # simplify hull to square
            hull = dicerfuncs.reduceHullPoints(hull, 4)

            # the entire thing
            rotatedRect = cv2.minAreaRect(hull)
            #
            #marginAdjust = 0.8
            marginAdjust = 0.9

            # mask it
            (img, mask) = self.applyForegroundExtractionMask_Square(img, mask, centroid, rotatedRect, marginAdjust)
            #rotatedRect2 = (rotatedRect[0], (rotatedRect[1][0] * marginAdjust, rotatedRect[1][1] * marginAdjust), rotatedRect[2])
            #color = 255
            #boxpoints = cv2.boxPoints(rotatedRect2)
            #boxpoints = boxpoints.astype(int)
            #cv2.fillConvexPoly(imgAnd, boxpoints, color)
            #mask = cv2.bitwise_and(imgAnd, mask)

            # other parameters
            self.maxDistanceContourAdd = maxside / 2.0
            self.maxDistanceContourAddFar = maxside / 2.0
            self.maxDistanceFaceCentroidAdd = maxside / 2



        # the mask we make may be dependent on self.shape
        elif (self.shape == "d10"):
            # circular shape
            radiusAll = min(centroid[0], centroid[1])
            radius = int(radiusAll / 1)

            # mask it
            (img, mask) = self.applyForegroundExtractionMask_Circle(img,mask,centroid,radius)
            #color = 255
            #cv2.circle(imgAnd, centroid, radius, color, thickness=-1)
            #mask = cv2.bitwise_and(imgAnd, mask)

            # other parameters we can be queried
            self.maxDistanceContourAdd = maxside / 40.0
            self.maxDistanceFaceCentroidAdd = maxside / 8.0

        elif (self.shape == "tri"):
            # circular shape
            radiusAll = min(centroid[0], centroid[1])
            radius = int(radiusAll / 1.1)

            # mask it
            (img, mask) = self.applyForegroundExtractionMask_Circle(img,mask,centroid,radius)

            # other parameters we can be queried
            self.maxDistanceContourAdd = maxside / 12.0
            self.maxDistanceFaceCentroidAdd = maxside / 8.0


        else:
            print "UNKNOWN DIE SHAPE PASSED: " + self.shape

        # see http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html
        return (img, mask)





    def applyForegroundExtractionMask_Circle(self, img, mask, centroid, radius):
        imgAnd = dicerfuncs.makeBinaryImageMaskForImg(mask)
        color = 255
        cv2.circle(imgAnd, centroid, radius, color, thickness=-1)
        mask = cv2.bitwise_and(imgAnd, mask)

        # and now mask crop the IMG itself!
        if (True):
            imgAnd = dicerfuncs.makeBinaryImageMaskForImg(mask)
            cv2.circle(imgAnd, centroid, radius+4, color, thickness=-1)
            img = cv2.bitwise_and(imgAnd, img)
            # actually crop? Note:this causes problems i think because of size changes to shape params
            if (False):
                x = centroid[0]
                y = centroid[1]
                r = radius+4
                xoffset = x-r
                yoffset = y-r
                img = img[yoffset:yoffset+2*r, xoffset:xoffset+2*r]
                mask = mask[yoffset:yoffset+2*r, xoffset:xoffset+2*r]
                #dicerfuncs.cvImgShow("DoubleCrop", img)

        return (img,mask)


    def applyForegroundExtractionMask_Square(self, img, mask, centroid, rotatedRect, marginAdjust):
        imgAnd = dicerfuncs.makeBinaryImageMaskForImg(mask)
        color = 255
        rotatedRect2 = (rotatedRect[0], (rotatedRect[1][0] * marginAdjust, rotatedRect[1][1] * marginAdjust), rotatedRect[2])
        boxpoints = cv2.boxPoints(rotatedRect2)
        boxpoints = boxpoints.astype(int)
        cv2.fillConvexPoly(imgAnd, boxpoints, color)
        mask = cv2.bitwise_and(imgAnd, mask)

        # mask crop image itself
        if (True):
            imgAnd = dicerfuncs.makeBinaryImageMaskForImg(mask)
            rotatedRect2 = (rotatedRect[0], (rotatedRect[1][0] * marginAdjust + 2, rotatedRect[1][1] * marginAdjust + 2), rotatedRect[2])
            boxpoints = cv2.boxPoints(rotatedRect2)
            boxpoints = boxpoints.astype(int)
            cv2.fillConvexPoly(imgAnd, boxpoints, color)
            img = cv2.bitwise_and(imgAnd, img)
            # actually crop? Note:this causes problems i think because of size changes to shape params
            if (False):
                # now lets find where we can crop
                x, y, w, h = cv2.boundingRect(boxpoints)
                img = img[y:y + h, x:x + w]
                mask = mask[y:y + h, x:x + w]

        return (img, mask)
    # -----------------------------------------------------------


    # -----------------------------------------------------------
    def extractForegroundColor(self, img, imgMask, maskHull, flag_debug):

        # first see if dieProfile has a quantizedColor foreground assigned
        if not (self.quantizedColorClusters is None):
            # we can use color plate extraction
            imgForeground = dicerfuncs.getQuantizedColorPlateByIndex(img, self.quantizedColorClusters,
                                                                     self.quantizedColorSpace,
                                                                     self.quantizedClosestColorIndex)
        else:
            # generic otsu
            # test
            if (False):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgForeground = dieextract.imgMaskedOtsu(imgGray, imgMask, maskHull, True, True, flag_debug)

        return imgForeground

    # -----------------------------------------------------------



    # -----------------------------------------------------------
    def cropImageIfAppropriate(self, img):
        (height, width) = img.shape[:2]

        # first, crop circle
        if (self.cropCircleR>0.0):
            # mask our circle crop
            x = int(self.cropCircleX * width)
            y = int(self.cropCircleY * height)
            r = int(self.cropCircleR * min(width, height))
            # create mask
            # make a mask image for it
            maskimg = dicerfuncs.makeBinaryImageMaskForImg(img)
            color = 255
            cv2.circle(maskimg, (x,y), r, color, thickness=-1, lineType=8, shift=0)
            # mask it
            img  = cv2.bitwise_and(img, img, mask=maskimg)
            # now we can create a crop based on this if we really want
            xoffset = x-r
            yoffset = y-r
            img = img[yoffset:yoffset+2*r, xoffset:xoffset+2*r]



        cropw = self.cropWidth
        croph = self.cropHeight
        if (cropw == 0):
            cropw = width
        if (croph == 0):
            croph = height
        if (cropw != width or croph != height):
            # do the crop
            xoffset = (width - cropw) / 2
            yoffset = (height - croph) / 2
            img = img[yoffset:height - yoffset, xoffset:width - xoffset]
        return img
    # -----------------------------------------------------------


    # -----------------------------------------------------------
    def getCamera(self):
        """Return previously created dieCamera or create new one."""
        if (self.diceCamera is None):
            # create new one
            self.diceCamera = dicecamera.DiceCamera.createCamera()
            pass
        return self.diceCamera
    # -----------------------------------------------------------





    # -----------------------------------------------------------
    def get_expectedFaceProbabilities(self):
        """For a normal dice this is just uniform probabilities."""

        # uniform
        expectedFaceProbabilities = [1.0/self.dieFaces] * self.dieFaces

        return expectedFaceProbabilities
    # -----------------------------------------------------------







    # -----------------------------------------------------------
    def doAutoRollIfAppropriate(self):
        """Trigger autorolling if appropriate."""
        if (not self.autoRoll):
            return

        # iterate the ui so we can update visuals before we do some sleeping during hardware spin commands
        c = dicerfuncs.checkForUserKey()
        k = c & 0xFF
        if k == ord('q'):
            return

        # spin
        autoRollHelper = self.getOrCreate_AutoRollHelper()
        autoRollHelper.spin()



    def getOrCreate_AutoRollHelper(self):
        if (self.autoRollHelper is None):
            # make it
            # create arduino relay version
            self.autoRollHelper = diecontrol.DieRollerHardware_ArduinoRelay()
            # init it
            global global_ArduinoComPort
            #print "comport = %s" % global_ArduinoComPort
            self.autoRollHelper.set_comPort(global_ArduinoComPort)
            # other init params
            #self.autoRollHelper.set_spinTimeMs(500)
            # init connect
            self.autoRollHelper.connect()

        # return it
        return self.autoRollHelper
    # -----------------------------------------------------------
