# !/usr/bin/python

# dicer
# version 1.0, 1/11/16
# mouser@donationcoder.com

# -----------------------------------------------------------
# imports
#
#from dieprofile import DieProfile
import dicerfuncs
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

# -----------------------------------------------------------










# -----------------------------------------------------------
def compareImageAgainstAnotherImageGetScore(dieProfile, img1, img2, flag_debug):

    if (True):
        # default brute force method
        return compareImageAgainstAnotherImageGetScore_Bf(img1,img2,flag_debug)

    if (flag_debug):
        dicerfuncs.cvImgShow("CompImg1", img1)
        dicerfuncs.cvImgShow("CompImg2", img2)

    # a dif compare method
    if (False):
        # imgreg compare
        # this does not seem to work at all
        import imgreg

        # preprocess to scale to same size and center
        (img1r,img2r) = preProcessImagesForCompare(img1,img2)

        if (flag_debug):
            dicerfuncs.cvImgShow("RCompImg1", img1r)
            dicerfuncs.cvImgShow("RCompImg2", img2r)

        # do imgreg compare
        im2, scale, angle, (t0, t1) = imgreg.similarity(img1r, img2r)
        if (flag_debug):
            print "did imgreg.similarity compare"
            dicerfuncs.cvImgShow("RTansform", im2)
            print "angle= %f scale=%f" % (angle, scale)
        score = 1.0
        bestAngle = 0.0
        bestImgDif = None
        return (score, bestAngle, img1, img1r, bestImgDif, img2, img2r)

    if (True):
        # die feature marker compare
        import diefeaturecomp

        # preprocess to scale to same size and center
        #(img1r,img2r) = preProcessImagesForCompare(img1,img2)
        if (False):
            img1r = cv2.resize(img1, (600, 600), interpolation=cv2.INTER_CUBIC)
            img2r = cv2.resize(img2, (600, 600), interpolation=cv2.INTER_CUBIC)
        else:
            img1r = img1
            img2r = img2

        score = diefeaturecomp.compareImageAgainstAnotherImageGetScore_Features(img1r, img2r, flag_debug)
        if (flag_debug):
            print "diefeaturecomp compare score = %f" % score
        return (score, 0.0, img1, img1r, None, img2, img2r)
# -----------------------------------------------------------




















# -----------------------------------------------------------
def preProcessImagesForCompare(img1,img2):
    #
    preprocessHints = ("")  # "dilate"
    postprocessHints = ("")  # "otsu"
    resizeHints = ()

    # parameters
    # option_resize from: largest|smallest|fixed
    option_resize = 'largest'
    option_resizeFixed = 64

    # first do any preprocessing BEFORE resizing (like eroding or dilating)
    # important that both images get SAME processing
    # since images may be different sizes this could be counter-productive
    img1 = imgCompareFilterProcess(img1, preprocessHints)
    img2 = imgCompareFilterProcess(img2, preprocessHints)

    # decide what size we want.. we could always normalize to a target, or resize to one of the img, but it should be square
    img1shape = img1.shape[:2]
    img2shape = img2.shape[:2]
    if (option_resize=='largest'):
        targetSize = max(img1shape[0],img1shape[1],img2shape[0],img2shape[1])
    elif (option_resize=='smallest'):
        targetSize = min(img1shape[0],img1shape[1],img2shape[0],img2shape[1])
    elif (option_resize=='fixed'):
        targetSize = option_resizeFixed
    else:
        raise "Unknown option_resize: %s" % option_resize

    # do resize of img1 (no rotation)
    img1r = imgRotateToSquareDestSize(img1, 0.0, targetSize)
    # do resize of img2 (no rotation)
    img2r = imgRotateToSquareDestSize(img2, 0.0, targetSize)

    return (img1r, img2r)
# -----------------------------------------------------------






















# -----------------------------------------------------------
def compareImageAgainstAnotherImageGetScore_Bf(img1, img2, flag_debug):

    preprocessHints = ("")  # "dilate"
    postprocessHints = ("")  # "otsu"
    resizeHints = ()
    #compareHints = ("erode",
    #                "dilate")  # try erode or denoise or open or binary (binary seems imp BUT the others mess with the threshold value used with it)
    compareHints = ()

    # ATTN: NEW -- now we compare img1 vs img2 and img2 vs img1 in case we get dif results, and take BEST
    result1 = imgCompareBf(img1, img2, preprocessHints, postprocessHints, resizeHints, compareHints, flag_debug)
    result2 = imgCompareBf(img2, img1, preprocessHints, postprocessHints, resizeHints, compareHints, flag_debug)
    if (result1[0]>result2[0]):
        result = result1
    else:
        result = result2

    # (score, bestAngle, img1a, img1r, bestImgDif, img2a, img2r) = result
    return result
# -----------------------------------------------------------




















































































# -----------------------------------------------------------
def imgCompareBf(img1, img2, preprocessHints, postprocessHints, resizeHints, compareHints, flag_debug):
    """Compare img1 and img2 and return a score from 0.0 to 1.0
	use filterHints to specify any preprocessing to do.
	images need not be same size
	but they should be B+W with white foreground (so dilation is predictable growing of edges)
	"""

    # parameters
    # option_resize from: largest|smallest|fixed
    option_resize = 'largest'
    option_resizeFixed = 64

    # first do any preprocessing BEFORE resizing (like eroding or dilating)
    # important that both images get SAME processing
    # since images may be different sizes this could be counter-productive
    img1 = imgCompareFilterProcess(img1, preprocessHints)
    img2 = imgCompareFilterProcess(img2, preprocessHints)

    # decide what size we want.. we could always normalize to a target, or resize to one of the img, but it should be square
    img1shape = img1.shape[:2]
    img2shape = img2.shape[:2]
    if (option_resize=='largest'):
        targetSize = max(img1shape[0],img1shape[1],img2shape[0],img2shape[1])
    elif (option_resize=='smallest'):
        targetSize = min(img1shape[0],img1shape[1],img2shape[0],img2shape[1])
    elif (option_resize=='fixed'):
        targetSize = option_resizeFixed
    else:
        raise "Unknown option_resize: %s" % option_resize


    # do resize of img1 (no rotation)
    img1r = imgRotateToSquareDestSize(img1, 0.0, targetSize)

    # post resize filtering
    img1r = imgCompareFilterProcess(img1r, postprocessHints)

    # now try to find the best matching angle
    # ATTN: we can try using a low fine grained angleInc and not do a 2-step refinement, or we can start with a higher angleInc, and do a second refinement, which seems to work pretty well
    angleInc = 5.0
    #angleInc = 1.0
    angleStart = 0.0
    angleEnd = 360.0
    angleFineTuneSteps = 10.0
    (score, bestAngle, bestImgDif, img2r) = imgCompareFindBestAngle(angleStart, angleEnd, angleInc, img1r, img2,
                                                                    targetSize, postprocessHints, compareHints)
    # if we wanted to get cute we could now hunt around the bestAngle with smaller incremenents, e.g.
    if (True):
        (score, bestAngle, bestImgDif, img2r) = imgCompareFindBestAngle(bestAngle - angleInc, bestAngle + angleInc,
                                                                        angleInc / angleFineTuneSteps, img1r, img2, targetSize,
                                                                        postprocessHints, compareHints)

    return (score, bestAngle, img1, img1r, bestImgDif, img2, img2r)



def imgCompareFindBestAngle(angleStart, angleEnd, angleInc, img1r, img2, targetSize, postprocessHints, compareHints):
    """Walk through angle range and find best match."""
    bestAngle = None
    bestImgDif = None
    maxScore = None
    angle = angleStart
    while (angle <= angleEnd):
        (score, imgDif, img2r) = imgCompareBitwise(img1r, img2, targetSize, angle, postprocessHints, compareHints)
        # print "Score is: %f" % score
        if (score > maxScore) or (maxScore is None):
            # new best
            maxScore = score
            bestAngle = angle
            bestImgDif = imgDif
            bestImg2r = img2r
        # advance angle
        angle = angle + angleInc
    return (maxScore, bestAngle, bestImgDif, bestImg2r)


def imgCompareBitwise(img1r, img2, targetSize, angle, postprocessHints, compareHints):
    """Do bitwise compare (no rotation that has been done by caller)."""

    # rotate img2 to angle
    img2r = imgRotateToSquareDestSize(img2, angle, targetSize)

    # any preprocessing AFTER resizing (like eroding or dilating)
    img2r = imgCompareFilterProcess(img2r, postprocessHints)

    # note that at this point we may have GRAYSCALE (not B+W image, due to resizing and processing)

    if (False):
        dicerfuncs.cvImgShow("IMG1R", img1r)
        dicerfuncs.cvImgShow("IMG2R", img2r)

    # now do bitwise compare
    imgDif = cv2.absdiff(img1r, img2r)

    if ("erode" in compareHints):
        # erode after diff to get rid of noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        imgDif = cv2.erode(imgDif, kernel, iterations=3)
    if ("dilate" in compareHints):
        # erode after diff to get rid of noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        imgDif = cv2.dilate(imgDif, kernel, iterations=1)
    if ("open" in compareHints):
        # erode after diff to get rid of noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        imgDif = cv2.morphologyEx(imgDif, cv2.MORPH_OPEN, kernel)
    if ("denoise" in compareHints):
        imgDif = cv2.fastNlMeansDenoising(imgDif, None, 3, 5, 7)
    if ("binary" in compareHints):
        threshlevel = 200
        # threshlevel = 64
        imgDif = cv2.threshold(imgDif, threshlevel, 255, cv2.THRESH_BINARY)[1]

    # now get AVERAGE diff value
    # print cv2.mean(imgDif)

    if (False):
        # orig
        if (True):
            # try to compute squared distance, to give higher punishment to bigger difs
            imgFDif = dicerfuncs.convertBgrTo32f(imgDif)
            imgFDif = np.square(imgFDif)
            imgVal = cv2.mean(imgFDif)[0]
            maxVal = 255.0 ** 2.0
            score = 1.0 - (float(imgVal) / maxVal)
        else:
            imgAvg = cv2.mean(imgDif)[0]
            # this will be in range of 0-255 so now scale to 0-1
            score = 1.0 - (float(imgAvg) / 255.0)
    else:
        # ATTN: 2/24/16 this makes a huge difference, at least on our d6s
        # what if we heavily weighted center?
        imgFDif = dicerfuncs.convertBgrTo32f(imgDif)
        imgFDif = np.square(imgFDif)
        imgVal = cv2.mean(imgFDif)[0]
        # center gets just as much weight
        height, width = imgFDif.shape[:2]
        xo = (width/3)
        yo = (height/3)
        imgDiffCenter = imgFDif[yo:yo + height-yo, xo:width-xo]
        imgValCenter = cv2.mean(imgDiffCenter)[0]
        imgVal += imgValCenter
        #maxVal = (255.0 ** 2.0) * 2
        #score = 1.0 - (float(imgVal) / maxVal)
        score = (-1.0) * (imgVal)

    # print "Returning score: %f" % score
    return (score, imgDif, img2r)


def imgCompareFilterProcess(img, processHints):
    """Run some filtering."""
    # important that both images get SAME processing
    # ATTN: todo
    if ("dilate" in processHints):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        img = cv2.dilate(img, kernel, iterations=1)
    # dicerfuncs.cvImgShow("DILATING",img)
    if ("erode" in processHints):
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        img = cv2.erode(img, kernel, iterations=1)
    if ("otsu" in processHints):
        otsuval, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # dicerfuncs.cvImgShow("otsu",img)

    return img


# -----------------------------------------------------------















# -----------------------------------------------------------
def compareImageAgainstFileGetScore(dieProfile, img, fpath, flag_debug):
    """Compare the image against the image in a file and return a score."""
    # dicerfuncs.debugprint("Comparing image against file: "+fpath)
    # (fileImage, fileMask) = loadPngSplitIntoImageAndMask(fpath)
    # dicerfuncs.cvImgShow(fpath,fileImage)

    fileImage = dicerfuncs.loadPngNoTransparency(fpath)
    # convert to bw
    fileImage = dicerfuncs.ensureGrayscale(fileImage, False)

    if (flag_debug and False):
        dicerfuncs.cvImgShow(fpath, fileImage)

    result = compareImageAgainstAnotherImageGetScore(dieProfile, img, fileImage, flag_debug)
    # (score, bestAngle, img1a, img1r, bestImgDif, img2a, img2r) = result
    return result
# -----------------------------------------------------------













# -----------------------------------------------------------
def imgRotateToSquareDest(img, angle):
    """We are given an image, we want to rotate it angle degrees, and center it in a new destinate image we will return
	such that the destination image is centered in a tight bounding square (black filled in around shorter dimension)."""
    # see http://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c
    height, width = img.shape[:2]
    centerPoint = (width / 2, height / 2)
    rot = cv2.getRotationMatrix2D(centerPoint, angle, 1)

    # we want to guess the size of targe destination, there are several ways to do it

    if (False):
        # super tight way (size will change based on angle)
        # rotated rect is ((x,y)(w,h),ANGLE)
        # now we compute bounding rectangle of this ROTATED image
        rotatedRect = ((centerPoint[0], centerPoint[1]), (width, height), angle)
        # 4 points making up the box
        box = cv2.boxPoints(rotatedRect)
        # is box == boundingRect of these 4 points
        boundingRect = cv2.boundingRect(box)
        destWidth = boundingRect[2]
        destHeight = boundingRect[3]
    else:
        # ATTN: this will result in uniform sizes IFF the passed image has been circumscribed to a minareaRect
        if (True):
            biggestEdge = max(height, width)
            # pythag theorom
            destBiggestEdge = math.sqrt(biggestEdge ** 2 + biggestEdge ** 2)
        else:
            # or is this sufficient
            destBiggestEdge = math.sqrt(width ** 2 + height ** 2)
        destWidth = destBiggestEdge
        destHeight = destBiggestEdge

    # print "box is"
    # print box
    # print "boundingrect is "
    # print boundingRect
    # print "rot is:"
    # print rot

    # now the clever part, adjusing rotation matrix to put it in center of this new destination (see page above for details)
    rot[0][2] = rot[0][2] + destWidth / 2.0 - centerPoint[0]
    rot[1][2] = rot[1][2] + destHeight / 2.0 - centerPoint[1]

    imgDest = cv2.warpAffine(img, rot, (int(destWidth), int(destHeight)), cv2.INTER_CUBIC)

    # NOW, do we need to crop
    flag_makeSquare = True
    imgDest = cropImageToContents(imgDest, flag_makeSquare)

    # test
    # imgDest = img

    return imgDest


def imgRotateToSquareDestSize(img, angle, edgeSize):
    """same as above but resize finished result."""
    img = imgRotateToSquareDest(img, angle)
    img = cv2.resize(img, (edgeSize, edgeSize), interpolation=cv2.INTER_CUBIC)
    return img


# -----------------------------------------------------------

# -----------------------------------------------------------
def imgCompositeRotateThroughAngles(img, edgeSize, angleInc):
    """Make a composite image of img at size edgeSize x edgeSize, rotating through 360 degrees.
	This is a debug function to help visualize rotation/normalization
	"""
    cellCount = int(360.0 / angleInc)
    imagesPerSide = int(math.sqrt(cellCount)) + 1
    offset = 5
    cellSize = edgeSize + (offset * 2)

    # compute destination size
    compositeWidth = imagesPerSide * cellSize
    compositeHeight = imagesPerSide * cellSize
    imgComposite = makeGrayscaleImage(compositeWidth, compositeHeight)
    angle = 0.0
    colIndex = 0
    rowIndex = 0
    while (angle < 360.0):
        # print angle
        # generate img
        imgCell = imgRotateToSquareDestSize(img, angle, edgeSize)
        # where to draw it
        x = (colIndex * cellSize) + offset
        y = (rowIndex * cellSize) + offset
        # draw it
        imgComposite[y:y + edgeSize, x:x + edgeSize] = imgCell
        # now advance
        colIndex = colIndex + 1
        if (colIndex >= imagesPerSide):
            colIndex = 0
            rowIndex = rowIndex + 1
        # advance angle
        angle = angle + angleInc
    # return it
    return imgComposite


# -----------------------------------------------------------





















# -----------------------------------------------------------
def guessLabelOfImage(img, dieProfile, subdir, flag_debug):
    """Try to guess the label of an image."""

    # file list (get or cache)
    fileList = dieProfile.get_fileList(True, subdir, dicerfuncs.get_filepattern_images())

    # loop and score a comparison of our image against every image file in our list
    scoreList = compareImageAgainstFileListGetScores(dieProfile, img, fileList, flag_debug)

    # sort (normalize if needed) all comparison scores so we have them in rank order
    scoreList = sortFileScores(scoreList)

    # when debugging, display top few die face and die directory(label) matches
    if (flag_debug):
        debugDisplayTopRankedScores(scoreList)

    # best
    (bestfpath, bestscore, bestAngle, img1a, img1r, bestImgDif, img2a, img2r, bestconfidence) = chooseBestScoringFile(scoreList,dieProfile)
    dicerfuncs.debugprint("Best match of %02.4f with confidence %02.4f : %s" % (
    bestscore * 100.0, bestconfidence * 100.0, dieProfile.calc_dieFaceIdFromPath(bestfpath)))

    # show something
    if (False):
        showTopNCompareResults(scoreList, 3, img1r, dieProfile)
    elif (False):
        dicerfuncs.cvImgShow("BestImg r2 " + bestfpath, img2a)
        dicerfuncs.cvImgShow("BestImg of " + bestfpath, img2r)
        dicerfuncs.cvImgShow("BestImg us " + bestfpath, img1r)
        dicerfuncs.cvImgShow("BestImgDif with " + bestfpath, bestImgDif)
    elif (True):
        dicerfuncs.cvImgShow("Camera normalized", img1r)
        dicerfuncs.cvImgShow("Match normalized", img2r)
        dicerfuncs.cvImgShow("Match Diff", bestImgDif)


# -----------------------------------------------------------




# -----------------------------------------------------------
def compareImageAgainstFileListGetScores(dieProfile, img, filelist, flag_debug):
    """Compare the image against all files in file list and return a score."""
    t0 = time.time()
    scorelist = list()
    for fpath in filelist:
        # compute score

        (score, bestAngle, img1a, img1r, bestImgDif, img2a, img2r) = compareImageAgainstFileGetScore(dieProfile, img, fpath, flag_debug)
        # add tuple
        scorelist.append((fpath, score, bestAngle, img1a, img1r, bestImgDif, img2a, img2r))

        if (flag_debug):
            print "Waiting for user to hit key to continue file list comparisons."
            while(True):
                k = cv2.waitKey(1) & 0xFF
                # quit on q
                if  k == ord(' ') or k==ord('q'):
                    break

    t1 = time.time()
    # print "Time to compare image against all prototypes: %0.5f seconds" % (t1-t0)
    return scorelist



def sortFileScores(scoreList):
    """Sort the list of scores."""
    # sort by second element of tuple (score)
    scoreList = sorted(scoreList, key=lambda x: x[1], reverse=True)
    return scoreList


def debugDisplayTopRankedScores(scoreList):
    """Display top ranked scores."""
    topn = 5
    topncount = min(topn, len(scoreList))
    dicerfuncs.debugprint("Top " + str(topn) + " matches:")
    for i in range(0, topncount):
        fpath = scoreList[i][0]
        score = scoreList[i][1]
        dicerfuncs.debugprint("   score for " + fpath + " is: " + str(score))


def chooseBestScoringFile(scoreList, dieProfile):
    """Choose top ranking file in sorted scoreList and resturn some measure of confidence."""
    bestfpath = ""
    score = 0.0
    bestconfidence = 0.0
    itemcount = len(scoreList)
    if len(scoreList) > 0:
        (bestfpath, score, bestAngle, img1a, img1r, bestImgDif, img2a, img2r) = scoreList[0]
    # now compute CONFIDENCE of this prediction
    # ATTN: how to
    # estimate confidence based on next best match
    bestClusterId = dieProfile.calc_dieFaceIdFromPath(scoreList[0][0])
    bestconfidence = 1.0
    for i in range(1,len(scoreList)):
        clusterId = dieProfile.calc_dieFaceIdFromPath(scoreList[i][0])
        if (clusterId != bestClusterId):
            bestconfidence = score - scoreList[i][1]
            #print "closest cluster is %s" % clusterId
            break
    return (bestfpath, score, bestAngle, img1a, img1r, bestImgDif, img2a, img2r, bestconfidence)


def showTopNCompareResults(scoreList, topn, img1r, dieProfile):
    """Show top n for comparison."""

    dicerfuncs.cvImgShow("Camera normalized", img1r)

    topncount = min(topn, len(scoreList))
    for i in range(0, topncount):
        (bestfpath, score, bestAngle, img1a, img1r, bestImgDif, img2a, img2r) = scoreList[i]
        fpath = scoreList[i][0]
        score = scoreList[i][1]
        dieId = dieProfile.calc_dieFaceIdFromPath(bestfpath)
        dicerfuncs.debugprint("TOP (%s) score: %02.4f" % (dieId, score))
        # dicerfuncs.cvImgShow("TopBestImg r2 "+bestfpath,img2a)
        dicerfuncs.cvImgShow("Top (%s) RotImg" % dieId, img2r)
        dicerfuncs.cvImgShow("Top (%s) Diff " % dieId, bestImgDif)


# -----------------------------------------------------------

































# -----------------------------------------------------------
def cropImageToContents(img, flag_makeSquare):
    """Crop image contents using contours."""

    if (img is None):
        return img

    # get contours
    imgCopy = dicerfuncs.copyCvImage(img)
    img2, contours, hierarchy = cv2.findContours(imgCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if (len(contours) == 0):
        return img

    allpoints = None
    if (True):
        unified = []
        allpoints = None
        # contourcount = len(contours)
        # status = np.zeros((contourcount,1))
        for i, cnt in enumerate(contours):
            if (True):
                if (allpoints is None):
                    allpoints = cnt
                else:
                    allpoints = np.vstack((allpoints, cnt))

    # dicerfuncs.debugprint("Done enumerating regions stage 2.")
    if (allpoints is None):
        return img

    # now find convex hull around points and return it
    hull = cv2.convexHull(allpoints)

    # now lets find where we can crop
    x, y, w, h = cv2.boundingRect(hull)

    # make square?
    # print "ONE x = %d y=%d w=%d h=%d" % (x,y,w,h)
    if (flag_makeSquare):
        if (h > w):
            dif = h - w
            dif2 = int(dif / 2)
            x = x - dif2
            w = h
            if (x < 0):
                x = 0
        elif (w > h):
            dif = w - h
            dif2 = int(dif / 2)
            y = y - dif2
            h = w
            if (y < 0):
                y = 0
    # print "TWO x = %d y=%d w=%d h=%d" % (x,y,w,h)

    imgCropped = img[y:y + h, x:x + w]

    # imgCropped = cv2.copyMakeBorder(imgCropped,2,2,2,2,cv2.BORDER_CONSTANT,(0,0,255))
    # cvImgShow("Bordered",imgCropped)

    return imgCropped
# -----------------------------------------------------------




