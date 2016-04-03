# !/usr/bin/python

# dicer
# version 1.0, 1/11/16
# mouser@donationcoder.com

"""
Commandline Use:
  -h, --help            show this help message and exit

"""

# -----------------------------------------------------------
# imports
#
from dieprofile import DieProfile
import diestats
import diecontrol
import diecompare
import diecolor
import dieextract
#
import sys
import os
import datetime
import random
import time
import math
import shutil
import copy
import re
# for open cv
import numpy as np
import cv2
import scipy
import scipy.stats
# sklearn (for clustering)
import sklearn.cluster
# plotting
import matplotlib
import matplotlib.pyplot as plt
# -----------------------------------------------------------


















# -----------------------------------------------------------
# wrapper to run tests

def captureAndProcessLoop(startMessage, processFunction, dieProfile, flag_debug, flag_smartChanges):
    """Run a live ongoing test of processing windows for debugging."""

    # parameters
    option_stableTimeBetweenSaves = 0.5
    option_stableTimeBeforeFirstSave = 2

    # globals for mouse callbacks
    global globalLiveCamImg, globalDieProfile, globalLiveCamWindowName, globalLiveCamX, globalLiveCamY

    #
    debugprint(startMessage + ". Hit 'q' key to quit...")

    # get capture device
    cap = dieProfile.getCamera()
    cap.allowCameraToFocus(False)

    # report time taken
    t0 = time.time()
    font = cv2.FONT_HERSHEY_SIMPLEX

    # for callbacks
    firstShow = True
    liveCamWindowName = "Live camera"
    globalLiveCamWindowName = liveCamWindowName

    # for making sure image is stable before we process
    lastImageSeen = None
    autoWantsSave = False
    processStartTime = time.time()
    saveStartTime = None

    while (True):
        # Capture frame-by-frame
        ret, rawframe = cap.read()

        # crop it?
        frame = dieProfile.cropImageIfAppropriate(rawframe)

        # show camera view with Frame Rate
        if (True or flag_debug):
            if (True):
                # let's write framerate on image (we resize here instead of in cmimgshow so our text is normal)
                #img = copyCvImage(rawframe)
                zoom = 1.0
                height, width = rawframe.shape[:2]
                img = cv2.resize(rawframe, (int(width * zoom), int(height * zoom)), interpolation=cv2.INTER_NEAREST)
                t1 = time.time()
                deltaT = t1 - t0
                t0 = t1
                frameRate = 1.0 / deltaT
                msg = "FPS: %02.2f" % frameRate
                cv2.putText(img, msg, (6, 30), font, 1, (0, 0, 255), 2)
            # show frame
            cvImgShow(liveCamWindowName, img)
            # set up callback?
            if (firstShow):
                userParam = None
                cv2.setMouseCallback(liveCamWindowName, liveCam_onMouse, userParam)
                firstShow = False
            # callback info
            globalLiveCamImg = frame
            globalDieProfile = dieProfile

        # let user quit with 'q'
        c = checkForUserKey()
        k = c & 0xFF

        if k == ord('q'):
            break


        if k == ord('p'):
            # color picker?
            if (True or dieProfile.get_colorPick() == True):
                # yes we want color picker
                diecolor.doColorPickForDie(globalLiveCamImg, dieProfile, img[globalLiveCamY, globalLiveCamX], True)
                autoWantsSave = True


        if (flag_smartChanges):
            # should we process it?
            if (lastImageSeen is None):
                isdiff = True
                lastImageSeen = frame
            else:
                isdiff = autoSaveImgIsDif(lastImageSeen, frame)

            if (isdiff):
                # ok the current image JUST changed, so update frame and maybe shcedule a process save
                #print "updating last seen image"
                lastImageSeen = frame
                # and get ready for next save (reset countdown)
                if ((time.time() - processStartTime) > option_stableTimeBeforeFirstSave):
                    autoWantsSave = True
                    saveStartTime = None
                # dont save again since its still changing
                continue

            # ok this image is same as the one we last saved, so do nothing
            #print "image unchanged since last save."
            if (not autoWantsSave):
                continue
            if (saveStartTime is None):
                #print "counting down to save time."
                saveStartTime = time.time()
                continue
            if ((time.time() - saveStartTime) < option_stableTimeBetweenSaves):
                #print "not enough time elapsed for stable save."
                continue

        # process it
        retv = processFunction(frame, dieProfile, k, flag_debug)
        if (not retv):
            break
        # clear save flag
        autoWantsSave = False
        # we need to update this too, so future compares get made against this
        lastImageSeen = frame

    # end cleanup
    cap.release()
    cv2.destroyAllWindows()


# -----------------------------------------------------------




# -----------------------------------------------------------
def liveCam_onMouse(event, x, y, flag, userParam):
    # user does something with mouse on livecam

    # globals for mouse callbacks
    global globalLiveCamImg, globalDieProfile, globalLiveCamWindowName, globalLiveCamX, globalLiveCamY

    # update mouse position tracking
    globalLiveCamX = x
    globalLiveCamY = y

    # print "liveCam_onMouse"
    # print userParam
    if (False):
        # show close up color picker window
        # cvImgShow(liveCameraWinowName, imgZoom)
        radius = 3
        color = (255, 0, 255)
        cv2.circle(globalLiveCamImg, (x, y), radius, color, 1)
        cvImgShow(globalLiveCamWindowName, globalLiveCamImg)
# -----------------------------------------------------------




















# -----------------------------------------------------------
# generic debug helpers

def debugprint(thing):
    """Just print out some info to debug console."""
    print thing


def sayHello():
    """Just say hello and show current date and time."""
    curtime = datetime.datetime.now()
    debugprint("Hello from dicer.py on %s" % curtime)


# -----------------------------------------------------------











# -----------------------------------------------------------
# constants
# IMPORTANT - see DieProfile class where we replace many of these

def getOutputDirectory():
    """Return base output directory, terminated with dir separator."""
    return ""


def getDiceDirectory():
    """Return base output directory, terminated with dir separator."""
    return getOutputDirectory() + "dice/"

def get_defaultBackgroundDirectory():
    return "background"

def get_backgroundFilename():
    return "background.png"
# -----------------------------------------------------------

















# -----------------------------------------------------------
# User interaction stuff

def waitForUserKey():
    c = cv2.waitKey(0)
    return c


def checkForUserKey():
    c = cv2.waitKey(1)
    return c


# -----------------------------------------------------------







# -----------------------------------------------------------
# Generic file functions

def myMakeDirPath(fpath):
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


def myMakeUniqueFilename(fdir, fname):
    """Get unique filename.
	See http://code.activestate.com/recipes/577200-make-unique-file-name/
	"""
    maxnum = 9999
    name, ext = os.path.splitext(fname)
    make_fn = lambda i: os.path.join(fdir, '%s_%02d%s' % (name, i, ext))
    for i in xrange(1, maxnum):
        uni_fn = make_fn(i)
        if not os.path.exists(uni_fn):
            return uni_fn
    return None


def UNUSED_buildRecursiveImageFileListFromDirectoryPattern(rootdir, repattern):
    """Give directory path (that might have wildcard) recursively find a list of all images, return as list."""
    filelist = list()
    for root, dirs, files in os.walk(rootdir):
        for name in files:
            if name.endswith((".png",)):
                fpath = os.path.join(root, name)
                filelist.append(fpath)
    return filelist


def clearContentsOfDirectory(subdir, dirPatternList):
    """Remove files in this directory."""
    #print "Asked to remove all contents of %s" % subdir
    filelist = list()
    for root, dirs, files in os.walk(subdir):
        for name in files:
            if name.endswith(dirPatternList):
                fpath = os.path.join(root, name)
                #print "asked to remove file: %s" % fpath
                os.remove(fpath)

    # now remove empty dirs
    for root, dirs, files in os.walk(subdir):
        for dir in dirs:
            fpath = os.path.join(root, dir)
            #print "asked to remove dir: %s" % fpath
            os.rmdir(fpath)



def copyFileToDir(sourcefile, destDir):
    """Copy file to destination directory."""
    myMakeDirPath(destDir)
    dst = destDir + "/" + os.path.basename(sourcefile)
    shutil.copy(sourcefile, dst)



def writeTextToFile(txt, fpath):
    file = open(fpath, "w")
    file.write(txt)
    file.close()


def removeBaseDir(basePath, fpath):
    """Remove leading basePath from fpath."""
    if (fpath.startswith(basePath)):
        retv = fpath[len(basePath):]
        if (retv[0]=='/' or retv[0]=='\\'):
            retv = retv[1:]
        return retv
    raise "Error in removeBaseDir, %s doesnt start with %s" % (fpath, basePath)
# -----------------------------------------------------------










# -----------------------------------------------------------
# Color conversion stuff

def convertBgrToHsvAndZeroV(img):
    """Convert and bgr image color format to hsv, and zero the v color channel
	The idea here is to remove shadow information for the purposes of doing background removal.
	"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # do we even HAVE to null it? or is just having it in hsv sufficient to make shadows appear closer when thresholding?
    img[:, :, 2] = 0
    return img


def convertBgrToHsv(img):
    """Convert and bgr image color format to hsv, and zero the v color channel
	The idea here is to remove shadow information for the purposes of doing background removal.
	"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img


def convertBgrToColorDistance(img):
    """Convert and bgr image color format to hsv, and zero the v color channel
	The idea here is to remove shadow information for the purposes of doing background removal.
	"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return img


def convertColorDistanceToBgr(img):
    """Convert and bgr image color format to hsv, and zero the v color channel
	The idea here is to remove shadow information for the purposes of doing background removal.
	"""
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    return img


def ensureGrayscale(img, flag_returncopy):
    """Return grayscale version if not grayscale."""
    if (len(img.shape) == 2):
        if (flag_returncopy):
            return copyCvImage(img)
        return img
    try:
        workingImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        if (flag_returncopy):
            return copyCvImage(img)
        return img
    return workingImage


def getColorSpaceEnumList():
    return ["BGR", "LAB", "HSV", "YCR"]


def convertBgrToColorSpace(img, colorSpaceEnum):
    """Convert bgr to color space provided by enum."""
    if (colorSpaceEnum == "LAB"):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    elif (colorSpaceEnum == "HSV"):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif (colorSpaceEnum == "YCR"):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    elif (colorSpaceEnum != "BGR"):
        raise "Unknown colorSpaceEnum."
    return img


def convertBgrToColorSpaceSinglePixel(pixelColor, colorSpaceEnum):
    """Convert bgr to color space provided by enum."""
    img = np.zeros((1, 1, 3), np.uint8)
    img[0, 0] = pixelColor
    retImg = convertBgrToColorSpace(img, colorSpaceEnum)
    return retImg[0][0]


def convertColorSpaceToBgr(img, colorSpaceEnum):
    """Convert bgr to color space provided by enum."""
    if (colorSpaceEnum == "LAB"):
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    elif (colorSpaceEnum == "HSV"):
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    elif (colorSpaceEnum == "YCR"):
        img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    elif (colorSpaceEnum != "BGR"):
        raise "Unknown colorSpaceEnum."
    return img


def convertColorSpaceToColorSpace(img, from_clusterColorSpace, to_clusterColorSpace):
    """Convert between color spaces."""
    # ATTN: we could check enums and do some DIRECT conversions later for better fidelity and speed, but for now we do a 2-step
    img = convertColorSpaceToBgr(img, from_clusterColorSpace)
    img = convertBgrToColorSpace(img, to_clusterColorSpace)
    return img


def convertRGBtoBGR(img):
    """Convert colorspace."""
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def convertBgrTo32f(img):
    return img.astype(float)


def imgInvert(img):
    """Simple invert."""
    img = (255 - img)
    return img


def ConvertBwGrayToBGR(img):
    """Just convert BW to BGR so we can overlay color on it."""
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


# -----------------------------------------------------------

















# -----------------------------------------------------------
# General purpose opencv image helper functions

def copyCvImage(img):
    """Really there is no cleaner way to make a copy of an image?"""
    return cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_REPLICATE)


def makeBinaryImageMaskForImg(img):
    """Make a binary same size image."""
    height, width = img.shape[:2]
    blank_image = np.zeros((height, width, 1), np.uint8)
    return blank_image


def makeGrayscaleImage(width, height):
    blank_image = np.zeros((height, width), np.uint8)
    return blank_image


def saveImageToPngFileWithMask(fpath, img, mask):
    """Given an img and a mask, save to png file."""
    b_channel, g_channel, r_channel = cv2.split(img)
    imga = cv2.merge((b_channel, g_channel, r_channel, mask))
    cv2.imwrite(fpath, imga)


def loadPngNoTransparency(fpath):
    """Standard png load without transparency"""
    img = cv2.imread(fpath)
    return img


def loadPngTransparency(fpath):
    """Open cv png read?"""
    img = cv2.imread(fpath, -1)
    return img


def loadPngSplitIntoImageAndMask(fpath):
    """Load png image from fpath and return as img, mask."""
    # imga = loadPngTransparency(fpath)
    imga = cv2.imread(fpath, -1)
    img = imga[:, :, 0:3]
    imgmask = imga[:, :, 3]
    return img, imgmask


def loadImgGeneric(fpath):
    """Generic load of image"""
    img = cv2.imread(fpath)
    return img


def saveImgGeneric(fpath, img, zoom = 1.0):
    """Generic write of image."""
    if (zoom < 1.0):
        height, width = img.shape[:2]
        targetwidth = int(width * zoom)
        targetheight = int(height * zoom)
        img = cv2.resize(img, (targetwidth, targetheight), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(fpath, img)


def resizeImageNicely(img, targetheight, targetwidth):
    """Make target size."""
    img_processed = cv2.resize(img, (targetwidth, targetheight), interpolation=cv2.INTER_CUBIC)
    return img_processed


def resizeExpandPlain(img, targetheight, targetwidth):
    """Make target size."""
    img_processed = cv2.resize(img, (targetwidth, targetheight), interpolation=cv2.INTER_NEAREST)
    return img_processed



def cropToCenterRectangle(img, centerPercent):
    """Crop image to certain percent of size."""
    h, w, c = img.shape

    if (centerPercent != 100):
        scanAdjust = ((float)(100 - centerPercent) / 100.0) / 2.0
        xoffset = int(scanAdjust * float(w))
        yoffset = int(scanAdjust * float(h))
        img = img[yoffset:h - yoffset, xoffset:w - xoffset]

    return img


def auto_canny(image, sigma=0.33):
    """from http://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
	"""
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged
    return edged


def drawHullOnImage(img, hull, flag_drawpoints):
    """Draw convex polygon on image."""

    # draw hull lines
    cv2.polylines(img, np.int32([hull]), True, 255, thickness=2)

    # now would be nice to draw POINTS
    if (flag_drawpoints):
        color = 255
        radius = 5
        for pt in hull:
            # print "%d x %d" % (pt[0][0],pt[0][1])
            cv2.circle(img, (pt[0][0], pt[0][1]), radius, color, 1)

    # draw centroid centroid of polygon/hull
    centroid = findHullMomentCentroid(hull)
    # centroid = findHullCentroidWalkPerimiter(hull)
    radius = 3
    color = (0, 0, 255)
    cv2.circle(img, centroid, radius, color, 1)


# -----------------------------------------------------------
























# -----------------------------------------------------------
# Background helper functions

def getBackgroundImage(dieProfile):
    """Load the background image and return it."""
    backgroundfilepath = getOutputDirectory() + dieProfile.get_backgroundFilepath()
    backgroundImage = loadImgGeneric(backgroundfilepath)
    return backgroundImage


def saveBackground(dieProfile):
    """Take a snapshot of camera with nothing on it for calibration purposes."""
    debugprint("Taking snapshot of camera image with nothing present to serve as background image.")

    # get capture device
    cap = dieProfile.getCamera()
    cap.allowCameraToFocus(True)

    # capture a frame
    ret, frame = cap.read()

    # now save it
    backgroundfilepath = dieProfile.get_backgroundFilepath()
    saveImgGeneric(backgroundfilepath, frame)

    # end cleanup
    cap.release()
# -----------------------------------------------------------























# -----------------------------------------------------------
def reduceHullPoints(hull, option_reduceHullPointCount):
    """Reduce hull polygon points to a target number."""
    # NOTE: the built-in fast approximation/simplification will not let us specify a target # of points in polygon
    # so we must hunt the right epsilon.
    # ATTN: todo, right a replacement c funciton to specifically reduce to target # directly
    startPoints = len(hull)
    targetPoints = option_reduceHullPointCount
    if (targetPoints >= startPoints):
        # already there
        return hull
    #
    minEp = 0.0
    maxEp = 100.0
    minimumEpDif = 0.001
    while (True):
        ep = (maxEp + minEp) / 2.0
        # print "Debug trying reduce ep = %f" % ep
        newhull = cv2.approxPolyDP(hull, epsilon=ep, closed=True)
        newPoints = len(newhull)
        # print " resulting points = %d to %d (target %d min = %f max =%f)" % (startPoints,newPoints,targetPoints,minEp,maxEp)
        # give up at some point
        if (maxEp - minEp < minimumEpDif):
            # print "Stopping due to convergence."
            break
        if (newPoints < targetPoints):
            # too few points in new one, so reduce epsilon
            maxEp = ep
        elif (newPoints > targetPoints):
            # too many points still, increase epsilon
            minEp = ep
        else:
            # found it!
            break
    return newhull


def reduceHullPointsSmart(hull, thresholdAreaLoss=0.1):
    """Reduce hull polygon points until doing so causes us to lose more than n% of area."""
    # NOTE: the built-in fast approximation/simplification will not let us specify a target # of points in polygon
    # so we must hunt the right epsilon.
    # ATTN: todo, right a replacement c funciton to specifically reduce to target # directly

    startArea = cv2.contourArea(hull)
    targetArea = float(startArea) * (1.0 - thresholdAreaLoss)

    # print "StartArea = %f   TargetArea=%f" % (startArea,targetArea)
    #
    minEp = 0.0
    maxEp = 100.0
    minimumEpDif = 0.001
    while (True):
        ep = (maxEp + minEp) / 2.0
        # print "Debug trying reduce ep = %f" % ep
        newhull = cv2.approxPolyDP(hull, epsilon=ep, closed=True)
        newArea = cv2.contourArea(newhull)
        newCount = len(newhull)
        # print "Trying ep of %f gives area=%f and count = %d (%f min %f max)" % (ep, newArea, newCount, minEp, maxEp)
        if (maxEp - minEp < minimumEpDif):
            # print "Stopping due to reduceHullPointsSmart convergence."
            break
        if (newArea < targetArea):
            # ok this ep is too large, its cutting off too much
            maxEp = ep
        elif (newArea > targetArea):
            minEp = ep
            largestGoodEp = ep
        else:
            break
    # use largest good one
    newhull = cv2.approxPolyDP(hull, epsilon=largestGoodEp, closed=True)
    return newhull


# -----------------------------------------------------------














# -----------------------------------------------------------
# see http://readmes.numm.org/looseleaf/dissect.py
def normalizeMinAreaRect(rect):
    """fix for fuxked up MinAreaRect."""
    ((cx, cy), (w, h), rot) = rect
    if rot < -45 and w < h:
        return ((cx, cy), (h, w), rot + 90)
    else:
        return rect


def imgRotateToMinAreaBoundingBox(hull, img, imgMask):
    # see http://felix.abecassis.me/2011/10/opencv-rotation-deskewing/
    # see http://stackoverflow.com/questions/11627362/how-to-straighten-a-rotated-rectangle-area-of-an-image-using-opencv-in-python
    # rotate the bounding box?
    rotatedRect = cv2.minAreaRect(hull)
    rotatedRect = normalizeMinAreaRect(rotatedRect)
    # very little documentation but return value is ((x,y)(w,h),ANGLE)
    # print rotatedRect
    angle = rotatedRect[2]
    box = cv2.boxPoints(rotatedRect)
    centerx = (box[0][0] + box[2][0]) / 2
    centery = (box[0][1] + box[2][1]) / 2
    center = (int(centerx), int(centery))

    if (not isRectSignifWiderThanTall(box)):
        angle += 90

    M = cv2.getRotationMatrix2D(center, angle, 1)
    height, width = img.shape[:2]
    newsize = (width, height)
    img = cv2.warpAffine(img, M, newsize, cv2.INTER_CUBIC)
    imgMask = cv2.warpAffine(imgMask, M, newsize, cv2.INTER_CUBIC)
    return (img, imgMask)


def isRectSignifWiderThanTall(box):
    """Return true if wider than tall."""
    width = calcPointPointDistance(box[0], box[1])
    height = calcPointPointDistance(box[1], box[2])
    if (height == 0):
        return True
    if (width / height > 1):
        return True
    return False


def imgDrawMinAreaBoundingBox(hull, img):
    # see http://felix.abecassis.me/2011/10/opencv-rotation-deskewing/
    # see http://stackoverflow.com/questions/11627362/how-to-straighten-a-rotated-rectangle-area-of-an-image-using-opencv-in-python
    # rotate the bounding box?
    rotatedRect = cv2.minAreaRect(hull)
    rotatedRect = normalizeMinAreaRect(rotatedRect)
    # very little documentation but return value is ((x,y)(w,h),ANGLE)
    # print rotatedRect
    angle = rotatedRect[2]
    box = cv2.boxPoints(rotatedRect)
    #

    box = np.int0(box)
    imgDrawn = copyCvImage(img)
    color = (0, 255, 0)
    if (False):
        cv2.drawContours(imgDrawn, [box], 0, color, 2)

    if (False):
        # center of box? not very useful
        centerx = (box[0][0] + box[2][0]) / 2
        centery = (box[0][1] + box[2][1]) / 2
        center = (int(centerx), int(centery))
        radius = 4
    # cv2.circle(imgDrawn, center, radius, color, 1)

    # centroid of polygon/hull
    centroid = findHullMomentCentroid(hull)
    # centroid = findHullCentroidWalkPerimiter(hull)
    radius = 4
    cv2.circle(imgDrawn, centroid, radius, color, 1)

    # fit line?
    if (False):
        # see http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
        # see http://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html#gsc.tab=0
        rows, cols = imgDrawn.shape[:2]
        [vx, vy, x, y] = cv2.fitLine(hull, cv2.DIST_L2, 0, 0.01, 0.01)
        lefty = int((-x * vy / vx) + y)
        righty = int(((cols - x) * vy / vx) + y)
        cv2.line(imgDrawn, (cols - 1, righty), (0, lefty), color, 2)

    if (False):
        vx = vx * -1
        # vy = vy * -1
        lefty = int((-x * vy / vx) + y)
        righty = int(((cols - x) * vy / vx) + y)
        cv2.line(imgDrawn, (cols - 1, righty), (0, lefty), color, 2)

    return imgDrawn


def findHullMomentCentroid(hull):
    """Find the center (centroid) of a convex hull."""
    M = cv2.moments(hull)
    m00 = M['m00']
    if (m00 == 0):
        m00 = 0.000001
    centroid = (int(M['m10'] / m00), int(M['m01'] / m00))
    return centroid


def findHullCentroidWalkPerimiter(hull):
    """Find the center (centroid) of a convex hull.
	This is my own idea"""
    pointCount = len(hull)
    perimiterLen = cv2.arcLength(hull, True)
    centerx = 0.0
    centery = 0.0
    for i, pt in enumerate(hull):
        # get connected point at end of line seg
        if i == pointCount - 1:
            pt2 = hull[0]
        else:
            pt2 = hull[i + 1]
        # midpoint of line segment
        print "pt1"
        print pt
        print "pt2"
        print pt2
        lineCenterx = (pt[0][0] + pt2[0][0]) / 2.0
        lineCentery = (pt[0][1] + pt2[0][1]) / 2.0
        # line segment length
        lineLen = math.sqrt((pt[0][0] - pt2[0][0]) ** 2 + (pt[0][1] - pt2[0][1]) ** 2)
        centerx = centerx + (lineCenterx * lineLen)
        centery = centery + (lineCentery * lineLen)
    # now divide by entire perim
    centerx = centerx / perimiterLen
    centery = centery / perimiterLen
    centroid = (int(centerx), int(centery))
    return centroid


# -----------------------------------------------------------









































# -----------------------------------------------------------
def testCandidateImage(img, imgMask, maskHull, dieProfile):
    # foreground extraction
    flag_debug = True
    imgForeground = extractForegroundFromImage(img, imgMask, maskHull, dieProfile, flag_debug)

    if (imgForeground is None):
        return

    imgAngles = imgCompositeRotateThroughAngles(imgForeground, 24, 1.0)
    cvImgShow("Die face angles", imgAngles)


# -----------------------------------------------------------




















































































# -----------------------------------------------------------
def computeDieFaceCenter(hull, dieProfile):
    """Given dieProfile and maskHULL, calculate the center point where die face label is most likely to be."""
    return dieProfile.computeFaceCenter(hull)


def calcPointPointDistance(pt1, pt2):
    """Distance between two 2d points."""
    dist = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
    # print "calcPointPointDistance (%d,%d) to (%d,%d) = %f" % (pt1[0],pt1[1],pt2[0],pt2[1],dist)
    return dist
# -----------------------------------------------------------












# -----------------------------------------------------------
def checkKeypressSaveImage(k, img, imgFull, dieProfile):
    """Let user label and save an image."""
    # lowercase to uppercase
    if (k >= ord('a') and k <= ord('z')):
        k -= 32
    # save index 1-9,A-Z
    if (k >= ord('1') and k <= ord('9')):
        labelImagesKeypress(dieProfile, k - ord('0'), img, imgFull)
    if (k >= ord('A') and k < ord('O')):
        labelImagesKeypress(dieProfile, (k - ord('A')) + 10, img, imgFull)
    if (k == ord('U')):
        labelImagesKeypress(dieProfile, 'unknown', img, imgFull)


def labelImagesKeypress(dieProfile, imageid, img, imgFull):
    """Save an image  with baselabel, imageid."""

    imagedir = dieProfile.get_faceImageDir("labeled/"+str(imageid))
    fpath = doSaveDieFaceImage(dieProfile, img, imgFull, imagedir)
    return fpath


def doSaveDieFaceImage(dieProfile, img, imgFull, imagedir):
    """Do the save."""

    fname = "image.png"
    fpath = dieProfile.makeUniqueFilename(imagedir, fname)

    if (fpath is None):
        return None

    print "In doSaveDieFaceImage with %s" % fpath

    # save extracted image
    saveImgGeneric(fpath, img)

    # save full camera image for reference
    #saveImgGeneric(dieProfile.changeExtension(fpath, "_cam.jpg"), imgFull, 0.5)
    saveImgGeneric(dieProfile.changeExtension(fpath, "_cam.png"), imgFull)
    return fpath
# -----------------------------------------------------------



































# -----------------------------------------------------------
def isBorderHandInView(imgDiff):
    """We want to do a sanity check on the imgDiff background to see if user's hand is in the way, so we can avoid heavy processing if so."""

    # ensure grayscale
    imgDiff = ensureGrayscale(imgDiff, False)

    # parameters
    thresholdColor = 15
    thresholdCountPercent = 0.05

    height, width = imgDiff.shape[:2]
    if (height < 4 or width < 4):
        return False;

    if isBorderHandInViewRect(imgDiff, 0, 0, width, 1, thresholdColor, thresholdCountPercent):
        return True
    if isBorderHandInViewRect(imgDiff, 0, 0, 1, height, thresholdColor, thresholdCountPercent):
        return True
    if isBorderHandInViewRect(imgDiff, 0, height - 1, width, height, thresholdColor, thresholdCountPercent):
        return True
    if isBorderHandInViewRect(imgDiff, width - 1, 0, width, height, thresholdColor, thresholdCountPercent):
        return True
    return False


def isBorderHandInViewRect(imgDiff, x, y, x2, y2, thresholdColor, thresholdCountPercent):
    """We want to do a sanity check on the imgDiff background to see if user's hand is in the way, so we can avoid heavy processing if so."""
    badCount = 0
    thresholdCount = thresholdCountPercent * float((x2 - x) * (y2 - y))
    for xp in range(x, x2):
        for yp in range(y, y2):
            if imgDiff[yp][xp] > thresholdColor:
                badCount = badCount + 1
                if (badCount > thresholdCount):
                    return True
    return False


# -----------------------------------------------------------





# -----------------------------------------------------------
def cvImgShow(windowName, img, zoom=1.0):
    """Show an image in a named window."""

    if (img is None):
        return

    # params
    #minWidth = 250
    #minHeight = 200
    minWidth = 150
    minHeight = 100


    # current size
    height, width = img.shape[:2]

    if (zoom != 1.0):
        # going to do some (fast) resizing
        img = cv2.resize(img, (int(width * zoom), int(height * zoom)), interpolation=cv2.INTER_NEAREST)
        height, width = img.shape[:2]

    # ensure min width so windows dont do funny garbage contents, pad with black
    addBorderH = 0
    addBorderV = 0
    if (width < minWidth):
        addBorderH = int((minWidth - width) / 2)
    if (height < minHeight):
        addBorderV = int((minHeight - height) / 2)
    if (addBorderH > 0 or addBorderV > 0):
        img = cv2.copyMakeBorder(img, addBorderV, addBorderV, addBorderH, addBorderH, cv2.BORDER_CONSTANT,
                                 value=(0, 0, 0))

    # show it
    cv2.imshow(windowName, img)


# -----------------------------------------------------------






























# -----------------------------------------------------------
def simplifyContours(contours, img):
    """Make all contours into minarearects."""
    height, width = img.shape[:2]
    for i, cnt in enumerate(contours):
        if (len(cnt) <= 4):
            continue
        # print "CONTOUR %d is:" %i
        # print cnt
        rotatedRect = cv2.minAreaRect(cnt)
        boxpoints = cv2.boxPoints(rotatedRect)
        boxpoints = boxpoints.astype(int)
        for j in range(0, 4):
            boxpoints[j][0] = min(boxpoints[j][0], width - 1)
            boxpoints[j][0] = max(0, boxpoints[j][0])
            boxpoints[j][1] = min(boxpoints[j][1], height - 1)
            boxpoints[j][1] = max(0, boxpoints[j][1])
        # cntSimplified = np.array( (np.array((np.array(boxpoints[0]))), np.array((np.array(boxpoints[1]))), np.array((np.array(boxpoints[2]))), np.array((np.array(boxpoints[3])))) )
        cntSimplified = np.array(([boxpoints[0]], [boxpoints[1]], [boxpoints[2]], [boxpoints[3]]))
        # print "now its:"
        # print cntSimplified
        contours[i] = cntSimplified
# -----------------------------------------------------------












































# -----------------------------------------------------------
def autoSaveImgIsDif(img1, img2):
    """Return true if these two images are different enought to trigger a new auto save (see dicer.py)."""

    # parameters
    # this needs to be set so that it only triggers different when die has been rolled
    option_scoreThreshold = 0.985

    # rescale them
    if (False):
        targetheight = 64
        targetwidth = 64
        img1r = resizeImageNicely(img1, targetheight, targetwidth)
        img2r = resizeImageNicely(img2, targetheight, targetwidth)
    else:
        img1r = img1
        img2r = img2

    #cvImgShow("comp1", img1r)
    #cvImgShow("comp2", img2r)

    # now do bitwise compare
    imgDif = cv2.absdiff(img1r, img2r)

    # score is how close (1 max, 0 min)
    if (False):
        # try to compute squared distance, to give higher punishment to bigger difs
        imgFDif = convertBgrTo32f(imgDif)
        imgFDif = np.square(imgFDif)
        imgVal = cv2.mean(imgFDif)[0]
        maxVal = 255 ** 2
        score = 1.0 - (float(imgVal) / maxVal)
    elif (True):
        imgAvg = cv2.mean(imgDif)[0]
        score = 1.0 - (float(imgAvg) / 255.0)

    #print "Img compare score dif = %2.04f" % score

    if (score < option_scoreThreshold):
        return True

    return False
# -----------------------------------------------------------





# -----------------------------------------------------------
def calcAllDieLabelsPossible(dieProfile, labeledFileList):
    """walk all labeled file list and build sorted list of all possible labels."""
    labelList = []
    for fpath in labeledFileList:
        dieLabel = dieProfile.calc_dieFaceIdFromPath(fpath)
        #print "dilabel = %s" % dieLabel
        if not dieLabel in labelList:
            labelList.append(dieLabel)
    labelList.sort()
    return labelList
# -----------------------------------------------------------









# -----------------------------------------------------------
def doFileLabel(dieProfile, unlabeledDir, labeledDir, flag_debug):
    """Label files in unlabeledDir according to labels in labeledDir."""

    # truncate for quicker testing?
    option_truncateCount = None
    # pairwise stats?
    option_showPairwiseStats = True

    # base path where report will be written
    reportBasePath = dieProfile.get_reportDirectory()

    # get file list (this will be the AUTO directory)
    unlabeledFileList = dieProfile.get_fileList(False, unlabeledDir, get_filepattern_images())
    labeledFileList = dieProfile.get_fileList(False, labeledDir, get_filepattern_images())

    # to report time taken
    t0 = time.time()

    # info
    print "Filelist sizes: %d unlabeled images and %d class images" % (len(unlabeledFileList),len(labeledFileList))


    # test truncate
    if (not option_truncateCount is None) and (len(unlabeledFileList)>option_truncateCount):
        print "WARNING: Truncating unlabeled file list to first %d items for faster testing." % option_truncateCount
        unlabeledFileList = unlabeledFileList[0:option_truncateCount]

    # storing results
    unlabeledResults = {}
    labeledClassResults = {}
    classCounts = {}
    pairwiseCounts = {}
    previousLabel = None

    allLabels = calcAllDieLabelsPossible(dieProfile, labeledFileList)

    # init
    for dieLabel in allLabels:
        labeledClassResults[dieLabel]=[]
        classCounts[dieLabel] = 0
        pairwiseCounts[dieLabel] = {}
        for nextDieLabel in allLabels:
            pairwiseCounts[dieLabel][nextDieLabel]=0


    # loop unlabeled files
    for i, unlabeledFile in enumerate(unlabeledFileList):
        print "Labeling %d of %d: %s" % (i,len(unlabeledFileList),unlabeledFile)
        # load file
        img = loadImgGeneric(unlabeledFile)
        # make grayscale
        img = ensureGrayscale(img, False)
        # loop and score a comparison of our image against every image file in our list
        scoreList = diecompare.compareImageAgainstFileListGetScores(dieProfile, img, labeledFileList, flag_debug)
        # sort (normalize if needed) all comparison scores so we have them in rank order
        scoreList = diecompare.sortFileScores(scoreList)
        # now get best match
        (bestfpath, bestscore, bestAngle, img1a, img1r, bestImgDif, img2a, img2r, bestconfidence) = diecompare.chooseBestScoringFile(scoreList, dieProfile)
        # now record it
        dieLabel = dieProfile.calc_dieFaceIdFromPath(bestfpath)
        #print "Best match for file %s was file %s with score %f" % (unlabeledFile, bestfpath, bestscore)
        #
        unlabeledResults[i] = (dieLabel, bestconfidence)
        # store
        labeledClassResults[dieLabel].append(i)
        classCounts[dieLabel] = classCounts[dieLabel] + 1
        # pairwise counts
        if (not previousLabel is None):
            pairwiseCounts[previousLabel][dieLabel] = pairwiseCounts[previousLabel][dieLabel] + 1
        # update previous label
        previousLabel = dieLabel


    t1 = time.time()
    print "File labeling took %f seconds" % (t1-t0)


    # helper list
    labeledFilesById = {}
    for i, labeledFile in enumerate(labeledFileList):
        dieLabel = dieProfile.calc_dieFaceIdFromPath(labeledFile)
        labeledFilesById[dieLabel] = labeledFile

    #print "unlabeledResults:"
    #print unlabeledResults
    #print "labeledClassResults:"
    #print labeledClassResults


    # visual report of classifications
    htmlReport = diestats.buildHtmlReportOfFileLabels(reportBasePath, dieProfile, unlabeledFileList, labeledFileList, unlabeledResults, labeledClassResults, labeledFilesById)
    htmlReport += "<br/><hr/><br/>"


    # statistics report
    imgsubname = dieProfile.get_uniqueLabel()
    htmlReportStats = diestats.buildHtmlReportStatistics(reportBasePath,dieProfile, classCounts, imgsubname)
    htmlReport += "Roll stats:<br/>"
    htmlReport += htmlReportStats

    # pairwise
    if (option_showPairwiseStats):
        htmlReport +="<br/><hr/><br/>";
        htmlReport += diestats.buildHtmlReportStatisticsPairwise(reportBasePath, dieProfile, pairwiseCounts)

    # write it out

    reportFPath = reportBasePath + "/" + "filelabel_report_" + dieProfile.get_uniqueLabel() + ".html"
    writeTextToFile(htmlReport, reportFPath)
    print "Saved report to %s" % reportFPath
# -----------------------------------------------------------







# -----------------------------------------------------------
def saveCopyOfBackground(dieProfile, subdir):
    """Save copy of background image into subdir."""
    img = getBackgroundImage(dieProfile)
    # now save it
    imagedir = dieProfile.get_faceImageDir(subdir)
    fpath = imagedir + '/' + get_backgroundFilename()
    saveImgGeneric(fpath, img)
# -----------------------------------------------------------


# -----------------------------------------------------------
def get_filepattern_images():
    """Regex for foreground images."""
    return "image_[\d]+\.png"


def get_filepattern_camphotos():
    """Regex for foreground images."""
    return "image_[\d]+_cam\.png"
# -----------------------------------------------------------





# -----------------------------------------------------------
def doExtractForegrounds(dieProfile, subdir, flag_debug):
    """Walk cam images in directory and extract foregrounds as if we were processing images live."""

    # get cam image file list
    camImageFileList = dieProfile.get_fileList(False, subdir, get_filepattern_camphotos())

    outimagedir = dieProfile.get_faceImageDir('unlabeled')

    # walk them
    pattern = "(image_[\d]+)_cam\.png"
    repl = "\\1.png"
    for i,fpath in enumerate(camImageFileList):
        print "Extracting from %d of %d: %s" % (i,len(camImageFileList),fpath)
        outfpath = re.sub(pattern, repl, fpath)
        #print "Extracting from '%s' to '%s'." % (fpath, outfpath)

        # load camera frame
        img = loadImgGeneric(fpath)

        # get foreground
        flag_isolateCenter = True
        imgForeground = dieextract.isolateAndExtractForegroundFromImage(img, dieProfile, flag_isolateCenter, flag_debug)
        if (imgForeground is None):
            continue

        # save extracted image
        saveImgGeneric(outfpath, imgForeground)
# -----------------------------------------------------------
