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
#from dieprofile import DieProfile
import diestats
import diecontrol
import dicerfuncs
#from dicerfuncs import *
import diecompare
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
def isolateAndExtractForegroundFromImage(img, dieProfile, flag_isolateCenter, flag_debug):
    """ extract die, then foreground."""
    (img, imgMask, maskHull) = simpleSingleForegroundObjectMaskAndCrop(img, dieProfile, flag_debug)
    if (maskHull is None):
        # couldn't find it
        return None
    imgForeground = extractForegroundFromImage(img, imgMask, maskHull, dieProfile, flag_isolateCenter, flag_debug)

    if (flag_debug):
        # show foreground
        dicerfuncs.cvImgShow('Isolated foreground', imgForeground)

    return imgForeground
# -----------------------------------------------------------




































# -----------------------------------------------------------
def extractForegroundFromImage(img, imgMask, maskHull, dieProfile, flag_isolateCenter, flag_debug):
    # ask die profile to get foreground (either generic otsumaked or color plate)

    if (img is None):
        return None

    if (dieProfile.get_dontThresholdForeground()):
        imgForeground = dicerfuncs.ensureGrayscale(img, False)
    else:
        imgForeground = dieProfile.extractForegroundColor(img, imgMask, maskHull, flag_debug)
        if (False and flag_debug and not (imgForeground is None)):
            dicerfuncs.cvImgShow("Foreground extraction", imgForeground)


    # further
    if (flag_isolateCenter):
        # skip processing if too big?
        if (False):
            maxAreaForForegroundCheck = 34000
            hullArea = cv2.contourArea(maskHull)
            if (hullArea > maxAreaForForegroundCheck):
                return None
        #
        (imgFace, faceMask, faceRegion) = isolateForegroundFace(imgForeground, imgMask, maskHull, dieProfile, flag_debug)
        if (False and flag_debug):
            dicerfuncs.cvImgShow("DIE FACE", imgFace)
        # crop
        (imgFace, faceMaskC, faceContourC) = cropImageGivenHull(imgFace, faceMask, faceRegion)
        if (False and flag_debug):
            dicerfuncs.cvImgShow("DIE FACE CROPPED", imgFace)
    else:
        # the entire foreground is the face
        imgFace = diecompare.cropImageToContents(imgForeground, True)
        #if (flag_debug):
        #    cvImgShow("WHOLE FOREGROUND CROPPED", imgFace)


    # dice wants denoisign at this stage?
    if (dieProfile.get_denoise()):
        dicerfuncs.cvImgShow('PreDenoisedFa', imgFace)
        #imgFace = cv2.fastNlMeansDenoising(imgFace, None, 3, 7, 21)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        imgFace = cv2.erode(imgFace, kernel, iterations=1)
        #imgFace = cv2.fastNlMeansDenoising(imgFace, None, 3, 5, 7)
        dicerfuncs.cvImgShow('DenoisedFa', imgFace)



    # return foreground face
    return imgFace


# -----------------------------------------------------------

























# -----------------------------------------------------------
def imgMaskedOtsu(img, imgMask, maskHull, flag_inverseLightBackground, flag_fillBackground, flag_debug):
    """Do otsu but only let it consider area inside imgMask.
	if flag_inverseLightBackground is True, try to make a white background black by inverting image."""
    # ok otsu thresholding has no notion of mask, so our approximation is to find a circumscribed bounding rectangle, and get otsu threshold value on THAT, then apply that manually to the entire image.
    flag_didinvert = False

    # grayscale
    imgGray = dicerfuncs.ensureGrayscale(img, True)

    if (False):
        # ok now test if its light and invert if so?
        if (flag_inverseLightBackground):
            (imgGray, flag_backgroundIsLight) = invertGrayscaleImageIfLight(imgGray, imgGray, imgMask)
        else:
            flag_backgroundIsLight = calcIsImageLight(imgGray, imgMask)

        # ok now for otsu below, we want the INVERSE of major color filled
        if True:
            flag_contrastBackgroundForOtsu = True
            if (flag_contrastBackgroundForOtsu):
                imgGray = fillMaskAreaWithLightDark(imgGray, imgMask, not flag_backgroundIsLight)
            else:
                imgGray = fillMaskAreaWithLightDark(imgGray, imgMask, flag_backgroundIsLight)



    # ok get circumscribed rectangle
    # there is no opencv function for this, so we have to code it ourselves
    # the most elegant way would be to get a minAreaRect rotated rectangle BOUNDING, (rotate it into normal angle) and then constrict bounds
    # but since this does not have to be so accurate we just avoid rotation for now
    # ATTN: flag_circumscribe can cause trouble if the region is too small
    option_thresholdRelativeInnerSize = 0.25
    (iw, ih) = imgGray.shape[:2]
    sizeOrig = iw * ih
    (x, y, w, h) = ComputeCircumScribedRectangleForRegion(maskHull, flag_circumscribe=True)
    sizeInner = w * h
    if (float(sizeInner) / float(sizeOrig)) < option_thresholdRelativeInnerSize:
        # too small an inner, we need to use whole thing!
        (x, y, w, h) = ComputeCircumScribedRectangleForRegion(maskHull, flag_circumscribe=False)
    if (w <= 0 or h <= 0):
        return None

    # ok crop to inner
    imgInner = imgGray[y:y + h, x:x + w]
    # print "inner is %d,%d to %d,%d" % (x,y,w,h)
    # print imgInner.shape
    # print "orig is"
    # print img.shape
    # test
    if (True and flag_debug):
        dicerfuncs.cvImgShow("Inner threshold", imgInner)

    # now ask otsu to COMPUTE the threshold inside circumscribed
    otsuval, imgOtsuMini = cv2.threshold(imgInner, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # dicerfuncs.cvImgShow("imgOtsuMini",imgOtsuMini)

    # now apply this to entire image
    retv, imgOtsu = cv2.threshold(imgGray, otsuval, 255, cv2.THRESH_BINARY)

    # otsu inner binary
    imgInner = imgOtsu[y:y + h, x:x + w]

    # now ask if background is light and invert if so
    # this ensures that the smaller # of pixels in our new BW otsu are LIGHT (ie white foreground)
    #(imgOtsu, flag_backgroundIsLight) = invertGrayscaleImageIfLight(imgOtsu, imgOtsu, imgMask)
    (imgOtsu, flag_backgroundIsLight) = invertGrayscaleImageIfLight(imgOtsu, imgInner, None)
    # fill background with reverse
    if (flag_fillBackground):
        #flag_backgroundIsLight = calcIsImageLight(imgGray, imgMask)
        imgOtsu = fillMaskAreaWithLightDark(imgOtsu, imgMask, not flag_backgroundIsLight)


    if (True and flag_debug):
        dicerfuncs.cvImgShow("imgOtsu Masked", imgOtsu)


    # print "imgotsu is:"
    # print imgOtsu.shape
    return imgOtsu


def invertGrayscaleImageIfLight(imgGray, imgInspect, imgMask):
    """Invert image if its lighter than dark
	see http://www.weheartcv.com/pixel-counting-color-statistics-part-1/
	see http://opencvpython.blogspot.com/2012/06/contours-3-extraction.html
	"""
    isImgLight = calcIsImageLight(imgInspect, imgMask)
    if (isImgLight):
        # majority light
        # dicerfuncs.debugprint("LIGHT BACKGROUND, INVERTING.")
        imgGray = dicerfuncs.imgInvert(imgGray)
        return (imgGray, True)
    return (imgGray, False)


def calcIsImageLight(imgGray, imgMask):
    # average color
    avgcolor = cv2.mean(imgGray, mask=imgMask)[0]
    # print "Average gray color: %d" % avgcolor
    # avge = np.average(image)
    # Scale result to the interval [0,1]
    avg01 = avgcolor / 255.0
    if (avg01 > 0.5):
        return True
    return False


def ComputeCircumScribedRectangleForRegion(maskHull, flag_circumscribe):
    """Get bounding rect than shirnk it."""
    x, y, w, h = cv2.boundingRect(maskHull)
    # print "boundingrect is %d,%d to %d,%d" % (x,y,w,h)
    if (not flag_circumscribe):
        return (x, y, w, h)
    return shrinkBoundingRectToCircumscribed(maskHull, x, y, w, h)


def shrinkBoundingRectToCircumscribed(maskHull, x, y, w, h):
    # now we are going to SHRINK it until all 4 corners fit
    # ul= np.array([0,0])
    # ur = np.array([0,0])
    # ll = np.array([0,0])
    # lr = np.array([0,0])
    # ul= np.zeros(2,dtype = np.int32)
    # ur = np.zeros(2,dtype = np.int32)
    # ll = np.zeros(2,dtype = np.int32)
    # lr = np.zeros(2,dtype = np.int32)

    if (maskHull is None):
        return (0,0,0,0)

    testPolygon = maskHull

    # print "test polygon is:"
    # print testPolygon

    while True:
        # ul[0]=x
        # ul[1]=y
        # ur[0]=x+w
        # ur[1]=y
        # ll[0]=x
        # ll[1]=y+h
        # lr[0]=x+2
        # lr[1]=y+h
        ul = (x, y)
        ur = (x + w - 1, y)
        ll = (x, y + h - 1)
        lr = (x + w - 1, y + h - 1)
        # ul = (y,x)
        # ur = (y,x+w)
        # ll = (y+h,x)
        # lr = (y+h,x+w)
        testval = (cv2.pointPolygonTest(testPolygon, ul, False) == 1) and (
        cv2.pointPolygonTest(testPolygon, ur, False) == 1) and (cv2.pointPolygonTest(testPolygon, ll, False) == 1) and (
                  cv2.pointPolygonTest(testPolygon, lr, False) == 1)
        if (testval):
            # print "matches at:"
            # print ul, ur, ll, lr
            return (x, y, w, h)
        # print "Testing %d,%d size %d,%d", (x,y,x+w,y+h)
        x = x + 1
        y = y + 1
        w = w - 2
        h = h - 2
        if (w <= 0 or h <= 0):
            return (0, 0, 0, 0)
    pass


def fillMaskAreaWithLightDark(img, imgMask, flag_backgroundIsLight):
    """Fill masked out area with light or dark."""
    # ATTN: unfinished
    imgMaskInverted = 255 - imgMask
    # imgMaskInverted = cv2.bitwise_not(imgMask)
    # dicerfuncs.cvImgShow("INVERTEST MASK",imgMaskInverted)
    # return img
    # print imgMask
    outImg = img.copy()
    # outImg = dicerfuncs.makeBinaryImageMaskForImg(img)
    if (flag_backgroundIsLight):
        cv2.bitwise_or(img, 255, outImg, mask=imgMaskInverted)
    else:
        cv2.bitwise_and(img, 0, outImg, mask=imgMaskInverted)
    return outImg


# -----------------------------------------------------------



























































# -----------------------------------------------------------
def simpleSingleForegroundObjectMaskAndCrop(img, dieProfile, flag_debug):
    """Try to crop out single simple foreground image"""

    if (img is None):
        return img, img, None

    # first pass mask of background
    flag_clearHsv = False
    flag_deNoise = False
    (backgroundMask, imgDiff) = imgRemoveBackgroundGetMask(img, dieProfile, flag_clearHsv, flag_deNoise, flag_debug)
    if (flag_debug):
        dicerfuncs.cvImgShow('Initial rough background mask', backgroundMask, zoom=0.5)

    if (img is None) or (backgroundMask is None):
        return img, img, None

    # check if hand in view and stop if so?
    if (False):
        isHandInView = isBorderHandInView(imgDiff)
        if (isHandInView):
            return img, img, None

    # crop to convex hull and clean mask
    option_maxpercentagesize = 25
    (imgCropped, maskCropped, convexHull) = imgCropConvexContentsGivenMask(img, backgroundMask,
                                                                           option_maxpercentagesize, flag_debug)
    if (False and flag_debug):
        dicerfuncs.cvImgShow('Convex contents crop', imgCropped)
        dicerfuncs.cvImgShow('Convex crop mask', maskCropped)

    return imgCropped, maskCropped, convexHull


# -----------------------------------------------------------




# -----------------------------------------------------------
def imgRemoveBackgroundGetMask(img, dieProfile, flag_clearHsv, flag_deNoise, flag_debug):
    """Remove the background from an image.
	See http://docs.opencv.org/master/db/d5c/tutorial_py_bg_subtraction.html#gsc.tab=0
	See https://techgimmick.wordpress.com/2015/03/11/background-subtraction-in-a-webcam-video-stream-using-emgucvopencv-wrapper-for-c/
	See http://stackoverflow.com/questions/10736933/frame-difference-noise
	See http://dsp.stackexchange.com/questions/11445/remove-background-from-image
	See http://www.thoughtfultech.co.uk/blog/simple-background-subtraction-for-loday.html
	See http://answers.opencv.org/question/17577/background-subtraction-from-a-still-image/
	See http://www.robindavid.fr/opencv-tutorial/chapter10-movement-detection-with-background.html
	See https://books.google.com/books?id=iNlOCwAAQBAJ&pg=PA179&lpg=PA179&dq=opencv+python+comparing+images+absdiff+smooth+threshold&source=bl&ots=iS-Ef_Toma&sig=K-aNUjZOEIOavkBsTOy6pPUoZCc&hl=en&sa=X&ved=0ahUKEwiG6MXPtKHKAhVHLmMKHf9UAB0Q6AEIPDAF#v=onepage&q=opencv%20python%20comparing%20images%20absdiff%20smooth%20threshold&f=false
	See http://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
	"""

    # for difficulty try red on white die (white background problem)

    workingImage = dicerfuncs.copyCvImage(img)
    backgroundImage = dicerfuncs.getBackgroundImage(dieProfile)
    backgroundImage = dieProfile.cropImageIfAppropriate(backgroundImage)

    # new idea to try to remove shadows, convert both hsv, then zero the v of both
    # this seems to help quite a lot, though it does kill dice that are same or similar (gray) color (hue) as background
    if (flag_clearHsv):
        # new idea
        workingImage = convertBgrToHsvAndZeroV(workingImage)
        backgroundImage = convertBgrToHsvAndZeroV(backgroundImage)
    # not sure thie is needed
    elif (True):
        workingImage = dicerfuncs.convertBgrToColorSpace(workingImage, "LAB")
        backgroundImage = dicerfuncs.convertBgrToColorSpace(backgroundImage, "LAB")




    # 2/12/16 - to get rid of spurious background
    if (True):
        workingImage = cv2.blur(workingImage,(10,10))
        backgroundImage = cv2.blur(backgroundImage,(10,10))




    # dif - subtract background from foreground
    imgDiff = cv2.absdiff(workingImage, backgroundImage);
    if (flag_debug):
        dicerfuncs.cvImgShow("Background Diff", imgDiff, zoom=0.5)

    img_mask = dicerfuncs.copyCvImage(imgDiff)

    # test 1/16/16
    img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
    if (False and flag_debug):
        dicerfuncs.cvImgShow("Background GrayMask", img_mask)

    if (False):
        # try fixed level threshold
        threshlevel = 15
        img_mask = cv2.threshold(img_mask, threshlevel, 255, cv2.THRESH_BINARY)[1]
        dicerfuncs.cvImgShow("Test thresh manual", img_mask)
        return img_mask


    # 2/13/16 - erode to try to get rid of circular dif errors at contianer color boundaries
    if (False):
        # this doesnt seem to be too needed on clean white background, nor does it do harm
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        img_mask = cv2.erode(img_mask, kernel, iterations=1)
        img_mask = cv2.dilate(img_mask, kernel, iterations=1)


    if (flag_deNoise):
        img_mask = cv2.fastNlMeansDenoising(img_mask, None, 3, 5, 7)

    img_mask = cv2.threshold(img_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # dicerfuncs.cvImgShow("Official Background MASK",img_mask)
    return (img_mask, imgDiff)

    # NOTHING USED BELOW HERE FOR NOW


    # test
    if flag_debug:
        if (flag_clearHsv):
            dicerfuncs.cvImgShow('HSV background diff', img_mask)
        else:
            dicerfuncs.cvImgShow('BGR background diff', img_mask)

    # colored denoising? (slow)
    if (False):
        img_mask = cv2.fastNlMeansDenoisingColored(img_mask, None, 10, 10, 7, 21)

    # now grayscale mask
    img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
    # test? (not very helpful)
    # dicerfuncs.cvImgShow('Background mask (grayscale)',img_mask)

    # erode (only if not using proper slow denoising function)
    if (not flag_deNoise):
        # this doesnt seem to be too needed on clean white background, nor does it do harm
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        img_mask = cv2.erode(img_mask, kernel, iterations=1)

    # dilate to fill in holes (only if not using proper slow denoising)
    # ATTN: for white die on white background, we need to run this dilation in order tocapture large convex area
    # but for clearer dice, it expands it too much and we get too much background
    # so ideally we should dynamically adjust this until we get it right
    # maybe by iterating until we have 1 main clearly separated hull
    if (not flag_deNoise):
        # this makes the mask much bigger than it should be
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        img_mask = cv2.dilate(img_mask, kernel, iterations=1)

    # formal gray denoise (slow)
    # denoising does make it less sensitive to threshold binary so we can make that lower min
    if (flag_deNoise):
        img_mask = cv2.fastNlMeansDenoising(img_mask, None, 3, 7, 21)
    # img_mask = cv2.fastNlMeansDenoising(img_mask,None,3,3,7)


    # threshold -- it's scary sensitive to this..
    # 13 seems pretty darn good, but we have seen some near-cutoffs with the d6 die that is bonewhite with red letter, in which case 5 works well but gives bigger background area
    threshlevel = 10
    threshlevel = 5
    if (flag_deNoise):
        threshlevel = 5
    img_mask = cv2.threshold(img_mask, threshlevel, 255, cv2.THRESH_BINARY)[1]

    return (img_mask, imgDiff)


# -----------------------------------------------------------







# -----------------------------------------------------------
def imgGetHullGivenMask(img_mask_in, option_maxpercentagesize):


    # combine/merge contours?
    # ATTN: 2/12/16 TRYING FALSE FOR TODAY TO SPEED UP
    flag_combineContours = False


    # how close a contour has to be to be merged in (50 seems to work pretty well); was 25 prior to feb 5
    option_distanceThreshold = 15
    # maximum regions and then we give up
    maxRegionsBeforeGiveUp = 5000
    # min size to merge
    option_contourMinSizeForMerge = 15

    # first lets find the contours on the mask
    img_mask = dicerfuncs.copyCvImage(img_mask_in)
    im2, contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # what we'd like to do is find the largest contour area, and then the convex region of it
    if contours is None:
        # no contours found
        return None

    imgProcessed = img_mask.copy()
    # cv2.drawContours(imgProcessed, contours, -1, (0,255,0), 3)
    # dicerfuncs.cvImgShow("newcontours",imgProcessed)

    # truncate if too many regions, sanity test
    if (len(contours) > maxRegionsBeforeGiveUp):
        # dicerfuncs.debugprint("Capping %d contours to top %d." % (len(contours),maxRegionsBeforeGiveUp))
        contours = contours[0:maxRegionsBeforeGiveUp]

    # too big test for messed up noise creating giant regions covering whole thing
    cutoff_areasize = float(img_mask.size) * (float(option_maxpercentagesize) / 100.0)

    # find largest contour
    contourindex_largest = 0
    countoursize_largest = 0
    contour_largest = None
    #
    for i, cnt in enumerate(contours):
        contoursize = cv2.contourArea(cnt)
        # hull = cv2.convexHull(cnt)
        # contoursize = cv2.contourArea(hull)
        if (contoursize > countoursize_largest and contoursize <= cutoff_areasize):
            countoursize_largest = contoursize
            contour_largest = cnt
            contourindex_largest = i

    # dicerfuncs.debugprint("Done enumerating regions stage 1.")

    # found a largest contour?
    if contour_largest is None:
        # no largest contour found
        return None

    # for speedier checks below
    bouncRectPointsLargest = counterFindBoundRectPoints(contour_largest)

    # ok now we actually MERGE all other NEARBY contours -- this makes us more forgiving and expansive in finding the global contour around main foreground object
    # See http://dsp.stackexchange.com/questions/2564/opencv-c-connect-nearby-contours-based-on-distance-between-them
    all_points = None
    if (flag_combineContours):
        unified = []
        allpoints = None
        # contourcount = len(contours)
        # status = np.zeros((contourcount,1))
        for i, cnt in enumerate(contours):
            # if (i == contourindex_largest) or (contourFindIfClose(cnt,contour_largest, option_distanceThreshold) == True):
            contoursize = cv2.contourArea(cnt)
            if (contoursize < option_contourMinSizeForMerge):
                continue
            # if (i == contourindex_largest) or (contourFindIfCloseFastApx(bouncRectPointsLargest,cnt, option_distanceThreshold) == True):
            if (i == contourindex_largest) or (
                contourFindIfClose(contour_largest, cnt, option_distanceThreshold) == True):
                if (allpoints is None):
                    allpoints = cnt
                else:
                    allpoints = np.vstack((allpoints, cnt))
    else:
        # ok just return convex hull around largest contour
        allpoints = contour_largest

    # dicerfuncs.debugprint("Done enumerating regions stage 2.")
    if (allpoints is None or len(allpoints) == 0):
        return None

    # now find convex hull around points and return it
    hull = cv2.convexHull(allpoints)

    # dicerfuncs.debugprint("Done enumerating regions stage 3.")

    return hull


# -----------------------------------------------------------








# -----------------------------------------------------------
def imgCropConvexContentsGivenMask(img, img_backgroundMask, option_maxpercentagesize, flag_debug):
    """Crop the contents out of the image.
	See http://opencvexamples.blogspot.com/2013/09/find-contour.html
	See http://docs.opencv.org/master/d4/d73/tutorial_py_contours_begin.html#gsc.tab=0
	See http://stackoverflow.com/questions/27192910/opencv-findcontours-loop-python
	See http://artsy.github.io/blog/2014/09/24/using-pattern-recognition-to-automatically-crop-framed-art/
	"""
    # given mask, extract image contents given mask

    # first compute likely hull of die
    hull = imgGetHullGivenMask(img_backgroundMask, option_maxpercentagesize)

    if hull is None:
        # hull not found
        return img, img_backgroundMask, None

    # debug show bounding box
    if (False and flag_debug):
        imgDrawn = imgDrawMinAreaBoundingBox(hull, img)
        dicerfuncs.cvImgShow("MinAreaRect", imgDrawn)

    flag_rotatebounding = False
    if (flag_rotatebounding):
        (img, img_backgroundMask) = imgRotateToMinAreaBoundingBox(hull, img, img_backgroundMask)
        # dicerfuncs.cvImgShow("ROTATED SHIT",imgRotated)
        # ATTN: TODO
        # and now regrab new hull given mask
        hull = imgGetHullGivenMask(img_backgroundMask, option_maxpercentagesize)
        if hull is None:
            # hull not found
            print "ERROR - hull lost after rotation."
            return img, img_backgroundMask, None

    # TEST - new hull simplification (try -1 for auto)
    option_reduceHullPointCount = None
    if (not (option_reduceHullPointCount is None)):
        if (option_reduceHullPointCount == -1):
            hull = reduceHullPointsSmart(hull)
        else:
            hull = reduceHullPoints(hull, option_reduceHullPointCount)

    img_processed = dicerfuncs.copyCvImage(img)

    if (flag_debug):
        # draw lines and contours
        # cv2.drawContours(img_processed, contours, -1, (0,255,0), 3)
        # now draw lines around main convex one
        dicerfuncs.drawHullOnImage(img_processed, hull, flag_drawpoints=True)
        dicerfuncs.cvImgShow("Convex hull ", img_processed, zoom=0.5)

    # make a mask image for it
    maskimg = dicerfuncs.makeBinaryImageMaskForImg(img)
    cv2.fillConvexPoly(maskimg, hull, 255);
    img_processed = cv2.bitwise_and(img, img, mask=maskimg)

    # now lets find where we can crop
    x, y, w, h = cv2.boundingRect(hull)

    # Crop the original image to the bounding rectangle
    img_processed_cropped = img_processed[y:y + h, x:x + w]
    mask_cropped = maskimg[y:y + h, x:x + w]

    # now we want to adjust hull to be valid in the cropped dimensions by subtracting x,y from each point in hull
    if (not (hull is None)):
        hull = adjustHullForCrop(hull, x, y)

    return img_processed_cropped, mask_cropped, hull


def adjustHullForCrop(hull, x, y):
    """After cropping adjust hull points."""
    newhull = hull.copy()
    for i, pt in enumerate(newhull):
        # print pt
        pt[0][0] = pt[0][0] - x
        pt[0][1] = pt[0][1] - y
    # newhull[i][0][0]=newhull[i][0][0]-x
    # newhull[i][0][1]=newhull[i][0][1]-x

    return newhull


# -----------------------------------------------------------





































# -----------------------------------------------------------
def maskImageGivenHull(img, hull):
    """Given a hull on an image, mask everything else to black, and CROP to smallest bounding rectangle."""

    if (hull is None):
        return (img, img)

    if (len(hull) < 4):
        return (img, img)

    # ok first of all, lets create a mask that is black outside hull
    imgMask = dicerfuncs.makeBinaryImageMaskForImg(img)
    cv2.fillConvexPoly(imgMask, hull, 255);
    img = cv2.bitwise_and(img, img, mask=imgMask)

    return (img, imgMask)


def cropImageGivenHull(img, imgMask, hull):
    """Given a hull on an image, mask everything else to black, and CROP to smallest bounding rectangle."""

    if (hull is None):
        return (img, img, hull)

    # ok first of all, lets create a mask that is black outside hull
    imgMask = dicerfuncs.makeBinaryImageMaskForImg(img)
    cv2.fillConvexPoly(imgMask, hull, 255);

    img = cv2.bitwise_and(img, img, mask=imgMask)

    if (False):
        # now minarea rotated rect
        (img, imgMask) = imgRotateToMinAreaBoundingBox(hull, img, imgMask)
        # and now regrab new hull given mask
        hull = imgGetHullGivenMask(imgMask, option_maxpercentagesize=100)

    if (hull is None):
        return (img, img, hull)

    if (True):
        # now get bounding rectangle and crop
        x, y, w, h = cv2.boundingRect(hull)
        img = img[y:y + h, x:x + w]
        imgMask = imgMask[y:y + h, x:x + w]
        # now we want to adjust hull to be valid in the cropped dimensions by subtracting x,y from each point in hull
        hull = adjustHullForCrop(hull, x, y)

    return (img, imgMask, hull)
# -----------------------------------------------------------

















































# -----------------------------------------------------------
def isolateForegroundFace(img, imgMask, maskHull, dieProfile, flag_debug):
    """We are passed a grayscale (bw really) img, where white is our foreground labels, and maskHull is our die shape.
	Now we want to extract the likely single face image.
	For d12+ this may mean extracting the region close to center, for d6 it may just be the whole thing.
	"""


    # params
    flag_denoise = False

    # default if we find nothing is just return what we were passed (ie everything is foreground)
    imgFace = img
    faceMask = imgMask
    faceRegion = maskHull
    faceCentroid = dicerfuncs.computeDieFaceCenter(maskHull, dieProfile)

    # denoise?
    if (flag_denoise):
        # img = cv2.fastNlMeansDenoising(img,None,3,5,7)
        img = cv2.fastNlMeansDenoising(img, None, 5, 7, 21)


    # first step would be to guess the shape and face info IFF not explicitly stated in dieProfile
    # ATTN: todo


    # perpare helper mask for foreground extraction, AND do some masking of main img to black out stuff that is NOT ALLOWED in foreground
    (img, extractionMask) = dieProfile.makeForegroundExtractionMask(img, faceMask, maskHull)


    # this get_useFullDieImage is also checkied in makeForegroundExtractionMask where it will return the image itself, usually unchanged
    if (dieProfile.get_useFullDieImage()):
        return (img, extractionMask, maskHull)


    if (dieProfile.get_diecenter()):
        # just use foreground focus dont choose contours
        # mask it
        img = cv2.bitwise_and(extractionMask, img)
        return (img, extractionMask, maskHull)


    # now find all regions (note we use RETR_LIST not RETR_EXTERNAL)
    imgRegions = dicerfuncs.copyCvImage(img)
    # imgDummy, contours, hierarchy = cv2.findContours(imgRegions,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    imgDummy, contours, hierarchy = cv2.findContours(imgRegions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours is None:
        return (imgFace, faceMask, faceRegion)

    # test - show contours on face
    if (flag_debug):
        imgProcessed = dicerfuncs.ConvertBwGrayToBGR(img)
        cv2.drawContours(imgProcessed, contours, -1, (255, 0, 255), 1)
        dicerfuncs.cvImgShow("Face contours", imgProcessed)

    if (False):
        # simplify all contours?
        simplifyContours(contours, imgMask)


    # test extraction mask
    if (flag_debug):
        imgTest = cv2.bitwise_xor(img, extractionMask)
        dicerfuncs.cvImgShow("FocusMask", imgTest)

    # now merge the APPROPRIATE face regions


    # start by finding the closest region to center point where we think the die face should be
    # this should be fairly fast
    contourIndex_closest = None
    contour_closest = None
    closestCentroid = None
    contourDistance_closest = 9999999.9
    for i, cnt in enumerate(contours):
        if (False):
            hull = cv2.convexHull(cnt)
            closestPoint = dicerfuncs.findHullMomentCentroid(hull)
            dist = calcPointPointDistance(faceCentroid, contourCentroid)
        else:
            (dist, closestPoint) = findClosestContourPointDistance(faceCentroid, cnt)
        if (dist < contourDistance_closest):
            # new closest
            contourDistance_closest = dist
            contourIndex_closest = i
            contour_closest = cnt
            closestCentroid = closestPoint

    if (contour_closest is None):
        # no closest contour -- im not sure how this can get here
        return (imgFace, faceMask, faceRegion)

    # do we need this?
    # hull_closest = cv2.convexHull(contour_closest)


    # merge nearby ones given closest starting contour
    # ATTN: there are dif ways we could compute "proximity"
    # a gross one would be centroid distance, but this is not ideal for merging die labels since we probably care most about the DOTS next to 6s and 9s, in whcih case proximity probably should be closest two points
    # this takes time unfortunately
    allpoints = contour_closest
    ignoreContours = list()
    ignoreContours.append(contourIndex_closest)

    while (True):
        didMerge = False
        for i, cnt in enumerate(contours):
            if (i in ignoreContours):
                continue

            # it might be nice to be able to do a QUICK reject
            # hull = cv2.convexHull(cnt)
            # contourCentroid = dicerfuncs.findHullMomentCentroid(cnt)
            # quickdist = calcPointPointDistance(closestCentroid, contourCentroid)
            # if (quickdist > quickRejectCentroidDistance):
            #	continue

            (accept, reject) = checkShouldMergeContour(dieProfile, allpoints, cnt, faceCentroid, closestCentroid,
                                                       extractionMask)
            if (accept):
                # merge it in
                didMerge = True
                ignoreContours.append(i)
                allpoints = np.vstack((allpoints, cnt))
            elif (reject):
                # permanently reject it
                didMerge = False
                ignoreContours.append(i)
            else:
                # leave it for next loop
                pass

        if (not didMerge):
            break

    faceContour = allpoints

    # is it already convex? if not i think we want convex
    faceContour = cv2.convexHull(faceContour)

    # ATTN: todo

    # now mask off this new face region
    (imgFace, faceMask) = maskImageGivenHull(img, faceContour)

    # now return it
    return (imgFace, faceMask, faceContour)


# -----------------------------------------------------------
























def checkShouldMergeContour(dieProfile, allpoints, cnt, faceCentroid, closestCentroid, extractionMask):
    """Check if we should merge a contour into our foreground list."""

    # ok so the question is, should a given contour be merged into what we consider the foreground contour set

    # a few ways to decide

    # there may be a distance (area) such that if the cnt under consideration has ANY point beyond that, we reject it
    # there may be a distance (area) such that if the cnt under consideration has its CENTROID point beyond that, we reject it
    # there may be a distance (area) such that if the cnt under consideration has its CLOSEST point beyond that, we reject it
    # these distances may be measure wrt the faceCentroid (typically center of die)
    #  or the closestCentroid (point on the first contour closest to faceCentroid)
    #  or the closest point int he CURRENT foreground (which keeps expanding)
    # other reasons we might reject merging a contour would be if adding it makes the foreground convex hull bigger than we expect

    # params
    minAreaReject = 0

    # ok lets check if any point is in the EXCLUDED area
    inExcluded = contourFindIfAnyPointExcluded(cnt, extractionMask)
    if (inExcluded):
        return (False, True)

    # too small?
    area = cv2.contourArea(cnt)
    if (area < minAreaReject):
        return (False, True)


    # some parameters
    # note that these parameters should be adjusted based on die size in pixels
    #	if (False):
    #		# good vaues for most
    #		dpMaxDistanceContourAdd = 8
    #		dpMaxDistanceContourReject = 36
    #		dpMaxDistanceCenterContourReject = 80
    #		quickRejectCentroidDistance = 50
    #	else:
    #		dpMaxDistanceContourAdd = 20
    #		dpMaxDistanceContourReject = 33
    #		dpMaxDistanceCenterContourReject = 40
    #		dpMinDistanceCenterContourReject = 40
    #		quickRejectCentroidDistance = 50

    dpMaxDistanceContourAdd = dieProfile.get_maxDistanceContourAdd()
    dpMaxDistanceContourAddFar = dieProfile.get_maxDistanceContourAddFar()
    dpMaxDistanceFaceCentroidAdd = dieProfile.get_maxDistanceFaceCentroidAdd()


    #isClose = contourFindIfClose(allpoints, cnt, dpMaxDistanceContourAdd)
    # ATTN new 2/25/16
    isClose = contourFindIfClose(allpoints, cnt, dpMaxDistanceContourAdd)

    if (isClose):
        # secondary check
        isClose = contourFindIfFarthestClose(allpoints, cnt, dpMaxDistanceContourAddFar)

    # 2/25/14 test
    #if (False and not isClose):
    if (not isClose):
        # it may be close enough to faceCentroid
        isClose = contourFindClosedPointIsCloseEnough(faceCentroid, cnt, dpMaxDistanceFaceCentroidAdd)


    # return
    return (isClose, False)


# -----------------------------------------------------------






# -----------------------------------------------------------
def contourFindIfAnyPointExcluded(cnt, extractionMask):
    """Return True if any point in contour is in a black maskedout area."""
    # height, width  = extractionMask.shape[:2]
    # print "contourFindIfAnyPointExcluded %d,%d" % (width, height)
    height, width = extractionMask.shape[:2]
    for i, pt in enumerate(cnt):
        # print "checking %d,%d" % (pt[0][0],pt[0][1])
        x = pt[0][0]
        y = pt[0][1]
        if (x >= width or y >= height):
            print "ERROR: contourFindIfAnyPointExcluded somehow x %d > %d or y %d > %d" % (x, width, y, height)
            return True
        if extractionMask[y, x] == 0:
            #print "Excluding contour point %d,%d." % (x,y)
            #print "TEST not excluding"
            #return False
            return True

    return False

# -----------------------------------------------------------



# -----------------------------------------------------------
def findClosestContourPointDistance(pt, cnt):
    """Find closest point in cnt to pt and return distance."""

    if (True):
        # fast built in function
        mindist = abs(cv2.pointPolygonTest(cnt, pt, True))
        # the function wont tell us closest, we could abandon getting closest and retunr centroid
        closestPoint =dicerfuncs.findHullMomentCentroid(cnt)
        return (mindist, closestPoint)

    mindist = 999999.9
    closestPoint = None
    for i, pt2 in enumerate(cnt):
        dist = calcPointPointDistance(pt, pt2[0])
        if (dist < mindist):
            mindist = dist
            closestPoint = pt2[0]
    return mindist, closestPoint


def contourFindIfClose(cnt1, cnt2, optionsDistanceThreshold):
    """Helper function True if two contours are 'close'.
	See http://dsp.stackexchange.com/questions/2564/opencv-c-connect-nearby-contours-based-on-distance-between-them"""
    # ATTN: THIS CAN BE SLOOOOOOOOOOOOOOW, figure out a fast appx
    for i, pt1 in enumerate(cnt1):
        pt = (pt1[0][0], pt1[0][1])
        (mindist, closestPoint) = findClosestContourPointDistance(pt, cnt2)
        if mindist < optionsDistanceThreshold:
            return True
    return False



def contourFindIfFarthestClose(cnt1, cnt2, optionsDistanceThreshold):
    """Helper function True if two contours are 'close'.
    This new function says, whats the farthest point in cn2 to its neartest point in cnt1
	See http://dsp.stackexchange.com/questions/2564/opencv-c-connect-nearby-contours-based-on-distance-between-them"""
    # ATTN: THIS CAN BE SLOOOOOOOOOOOOOOW, figure out a fast appx
    maxdistfound = -1
    for i, pt1 in enumerate(cnt2):
        pt = (pt1[0][0], pt1[0][1])
        (mindist, closestPoint) = findClosestContourPointDistance(pt, cnt1)
        if (mindist > maxdistfound):
            maxdistfound = mindist
    if maxdistfound < optionsDistanceThreshold and maxdistfound>=0:
        return True
    return False




def contourFindIfCloseOLD(cnt1, cnt2, optionsDistanceThreshold):
    """Helper function True if two contours are 'close'.
	See http://dsp.stackexchange.com/questions/2564/opencv-c-connect-nearby-contours-based-on-distance-between-them"""
    # ATTN: THIS CAN BE SLOOOOOOOOOOOOOOW, figure out a fast appx
    for i, pt1 in enumerate(cnt1):
        for j, pt2 in enumerate(cnt2):
            dist = calcPointPointDistance(pt1[0], pt2[0])
            if dist < optionsDistanceThreshold:
                return True
    return False


def contourFindIfFar(cnt1, cnt2, optionsDistanceThreshold):
    """Helper function True if the farthest away point on cn2, to the closest point on cn1, is beyond some value.
	See http://dsp.stackexchange.com/questions/2564/opencv-c-connect-nearby-contours-based-on-distance-between-them"""
    # ATTN: this can be re-written quicker using pointPolygonTest (http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html)
    # testing from cn2 to cnt1
    # ATTN: THIS CAN BE SLOOOOOOOOOOOOOOW, figure out a fast appx
    # print "checking cnt1"
    # print cnt1
    # print "checking cnt2"
    # print cnt2
    mindist = 999999.99
    # find the pt1 which is CLOSEST to the FARTHEST pt2
    for i, pt1 in enumerate(cnt1):
        # find the point on cn2 FARTHEST away from pt1
        maxdist_tothispoint = 0
        for j, pt2 in enumerate(cnt2):
            dist = calcPointPointDistance(pt1[0], pt2[0])
            if dist > maxdist_tothispoint:
                maxdist_tothispoint = dist
        # ok now we have maxdist to this point
        if (maxdist_tothispoint < mindist):
            mindist = maxdist_tothispoint
    if (mindist > optionsDistanceThreshold):
        # the closest point on cnt1 to the farthest point on cnt2 is bigger than threshold
        return True
    return False


def contourFindIfFarFromPoint(pt, cnt, optionsDistanceThreshold):
    maxdist_tothispoint = 0
    for j, pt2 in enumerate(cnt):
        dist = calcPointPointDistance(pt, pt2[0])
        if dist > maxdist_tothispoint:
            maxdist_tothispoint = dist
    if (maxdist_tothispoint > optionsDistanceThreshold):
        # the closest point on cn1 to the farthest point on cnt2 is bigger than threshold
        return True
    return False


def contourFindClosedPointIsCloseEnough(pt, cnt, optionsDistanceThreshold):
    # we want to know if the closest point
    (mindist, closestPoint) = findClosestContourPointDistance(pt, cnt)
    if (mindist < optionsDistanceThreshold):
        return True
    return False

    minist_tothispoint = 99999
    for j, pt2 in enumerate(cnt):
        dist = calcPointPointDistance(pt, pt2[0])
        if dist < minist_tothispoint:
            minist_tothispoint = dist
    if (minist_tothispoint < optionsDistanceThreshold):
        # the closest point on cn1 to the farthest point on cnt2 is bigger than threshold
        return True
    return False


def contourFindIfClose_OLDUNUSED(cnt1, cnt2, optionsDistanceThreshold):
    """Helper function True if two contours are 'close'.
	See http://dsp.stackexchange.com/questions/2564/opencv-c-connect-nearby-contours-based-on-distance-between-them"""
    # ATTN: THIS CAN BE SLOOOOOOOOOOOOOOW, figure out a fast appx
    row1, row2 = cnt1.shape[0], cnt2.shape[0]
    for i in xrange(row1):
        for j in xrange(row2):
            dist = np.linalg.norm(cnt1[i] - cnt2[j])
            if abs(dist) < optionsDistanceThreshold:
                return True
            elif i == row1 - 1 and j == row2 - 1:
                return False
    return false


def contourFindIfCloseFastApx(bouncRectPoints1, cnt2, optionsDistanceThreshold):
    """Helper function True if two contours are 'close'.
	See http://dsp.stackexchange.com/questions/2564/opencv-c-connect-nearby-contours-based-on-distance-between-them"""
    # ATTN: we are currently asking if ANY point is nearby.. perhaps it would be smarter to ask if the FARTHEST point is nearby, OR if the CENTROID point is nearby any of our conrners..
    bouncRectPoints2 = counterFindBoundRectPoints(cnt2)
    # ATTN: this is not the smartest thing, and also at least do MinAreaRect not square boundingRect
    for i in bouncRectPoints1:
        for j in bouncRectPoints2:
            # dist = np.linalg.norm(bouncRectPoints1-bouncRectPoints2)
            dist = math.sqrt((i[0] - j[0]) ** 2 + (i[1] - j[1]) ** 2)
            if abs(dist) < optionsDistanceThreshold:
                return True
    return False


def counterFindBoundRectPoints(cnt):
    """Given a contour, return 4 points in a list representing the locations of 4 bounding box rectangle points."""
    x, y, w, h = cv2.boundingRect(cnt)
    pointlist = ([x, y], [x + w, y], [x, y + h], [x + w, y + h])
    return pointlist


# -----------------------------------------------------------









