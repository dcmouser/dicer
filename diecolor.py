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
import dicerfuncs
import dieextract
#from dicerfuncs import *
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
def testColorQuantization(img, imgMask, maskHull, dieProfile, flag_debug):
    """Test color quantization."""
    # how many color clusters for k-means (4 seems pretty good)
    numColors = 15
    # color compression
    numColorsMergeTo = 999
    # this says to scan approximately X% of center rows and columns (so 50 would skip 25% from top, bottom, left, rigth, scanning an area width/2 x height/2
    # on some dice, setting this to like 50 instead of 100 really helps picking up font color with small # of colors
    scanPercent = 100
    # resize to standard small size to make color scanning faster? (0 means no target pixel size i.e. dont shrink)
    # option_pixelCountGoal = 0
    option_pixelCountGoal = 200 * 200
    # convert to color format for better/different color distances (BGR, HSV, LAB, YCR)
    # my experience HSV does not do so great (see gold on red die is noisy); LAB and YCR seem pretty good; test with small numColors
    option_clusterColorSpace = "LAB"
    # ignore prior mask?

    # test, use acanny edges as mask
    # imgCanny = auto_canny(imgMask, sigma = .044)
    if (False):
        imgCanny = cv2.Canny(img, 1, 244)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        imgCanny = cv2.dilate(imgCanny, kernel, iterations=1)
        dicerfuncs.cvImgShow("CANNY", imgCanny)
        imgMask = imgCanny

    # imgPriorMask = None
    colorClusters = imgFindDominantColors(img, imgMask, numColors, scanPercent, option_pixelCountGoal,
                                          option_clusterColorSpace, flag_debug)

    if (numColorsMergeTo < len(colorClusters)):
        # compress color clusters further with merging
        colorClusters = mergeColorClusters(colorClusters, numColorsMergeTo, option_clusterColorSpace)
        print "Reduced to %d clusters." % len(colorClusters)

    if (not (colorClusters is None)):
        quantizedImage = quantizeColorsInImage(img, colorClusters, option_clusterColorSpace)
        if (flag_debug):
            dicerfuncs.cvImgShow("Quantized colors", quantizedImage)
            #
            # testCandidateImage(quantizedImage, imgMask, maskHull, dieProfile)
            #
            showQuantizedColorPlates(img, colorClusters, option_clusterColorSpace)

    # done
    return colorClusters


# -----------------------------------------------------------











































# -----------------------------------------------------------
def testColorQuantizationUnderPercent(img, imgMask, underPercent, flag_debug):
    """Test color quantization."""
    # how many color clusters for k-means (4 seems pretty good)
    # numColors = 2
    # color compression
    numColorsMergeTo = 999
    # this says to scan approximately X% of center rows and columns (so 50 would skip 25% from top, bottom, left, rigth, scanning an area width/2 x height/2
    # on some dice, setting this to like 50 instead of 100 really helps picking up font color with small # of colors
    scanPercent = 70
    # resize to standard small size to make color scanning faster? (0 means no target pixel size i.e. dont shrink)
    # option_pixelCountGoal = 0
    option_pixelCountGoal = 200 * 200
    # option_pixelCountGoal = 75*75
    # convert to color format for better/different color distances (BGR, HSV, LAB, YCR)
    # my experience HSV does not do so great (see gold on red die is noisy); LAB and YCR seem pretty good; test with small numColors
    option_clusterColorSpace = "LAB"

    maxNumColors = 8
    for numColors in range(2, maxNumColors):
        colorClusters = imgFindDominantColors(img, imgMask, numColors, scanPercent, option_pixelCountGoal,
                                              option_clusterColorSpace, flag_debug)
        if (colorClusters is None):
            continue
        (smallestWeightPercentage, smallestWeightIndex) = calcSmallestPercentageWeight(colorClusters)
        if (smallestWeightPercentage < underPercent):
            break

    if (numColorsMergeTo < len(colorClusters)):
        # compress color clusters further with merging
        colorClusters = mergeColorClusters(colorClusters, numColorsMergeTo, option_clusterColorSpace)
        print "Reduced to %d clusters - best is %d." % (len(colorClusters), smallestWeightIndex)

    if (not (colorClusters is None)):
        quantizedImage = quantizeColorsInImage(img, colorClusters, option_clusterColorSpace)
        if (flag_debug):
            dicerfuncs.cvImgShow("Quantized colors", quantizedImage)
            showQuantizedColorPlateByIndex(img, colorClusters, option_clusterColorSpace, smallestWeightIndex)

    # done
    return colorClusters


# -----------------------------------------------------------















# -----------------------------------------------------------
# Functions for finding dominant colors, etc.

def imgFindDominantColors(img, imgPriorMask, numColors, scanPercent, option_pixelCountGoal, option_clusterColorSpace,
                          flag_debug):
    """Scan the image (the centermost scanPercent) and find the numColors ranked dominant colors.
	Note that you NEED to give numColors bigger than one because this really tiles the color space with  numColors nearest neighbors, so
	passing numColors = 1 would just give you the AVERAGE (or median?) color.
	See	http://giusedroid.blogspot.com/2015/04/using-python-and-k-means-in-hsv-color.html
	See http://www.alanzucconi.com/2015/05/24/how-to-find-the-main-colours-in-an-image/
	See http://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
	"""

    # convert to hsv space?
    # ATTN: should we do this before or after resize?
    img = dicerfuncs.convertBgrToColorSpace(img, option_clusterColorSpace)

    # shrink first?
    if (option_pixelCountGoal > 0):
        img = imgShrinkToPixelCount(img, option_pixelCountGoal)
        imgPriorMask = imgShrinkToPixelCount(imgPriorMask, option_pixelCountGoal)

    # ok convert image into flat list of pixels (with options to only scan some pixels and ignore others)
    rawpixels = imgFlattenRawPixels(img, imgPriorMask, scanPercent)
    if flag_debug:
        dicerfuncs.debugprint("Flattened pixel count: %d " % len(rawpixels))

    if (len(rawpixels) < 4):
        # not enough
        return None

    colorClusters = imgClusterColorsFromPixelList(rawpixels, numColors)
    if flag_debug:
        dicerfuncs.debugprint("Ranked color clusters:")
        dicerfuncs.debugprint(colorClusters)

    imgColorHistogram = imgClusterColorsFromPixelList_drawHistogramFromColorClusters(colorClusters,
                                                                                     option_clusterColorSpace)
    if flag_debug:
        dicerfuncs.cvImgShow("Color cluster histogram", imgColorHistogram)

    return colorClusters


def imgFlattenRawPixels(img, imgPriorMask, scanPercent):
    """Flatten img into vector of pixels.
	This is one fast line of code IFF scanPercent is 100 and no mask, but could be SLOW if not."""
    h, w, c = img.shape

    if (False) or (scanPercent == 100 and imgPriorMask is None):
        pixelvector = img.reshape((h * w, c))
    else:
        # walk it
        # from 0 to 0.5 measure of how offset each dimension
        scanAdjust = ((float)(100 - scanPercent) / 100.0) / 2.0
        xoffset = int(scanAdjust * float(w))
        yoffset = int(scanAdjust * float(h))
        # python list array
        pixelist = []
        for y in range(0 + yoffset, h - yoffset):
            for x in range(0 + xoffset, w - xoffset):
                if (not (imgPriorMask is None)):
                    # check (binary) mask
                    maskpixel = imgPriorMask[y][x]
                    if (maskpixel == 0):
                        # its masked
                        continue
                # add pixel
                pixel = img[y][x]
                pixelist.append(pixel)
        # now  build numpy pixel vector array
        pixelvector = np.array(pixelist)

    return pixelvector


def imgClusterColorsFromPixelList(pixelList, numClusters):
    """Given a numpy array of rawpixels, do clustering.
	See	http://giusedroid.blogspot.com/2015/04/using-python-and-k-means-in-hsv-color.html
	See http://www.alanzucconi.com/2015/05/24/how-to-find-the-main-colours-in-an-image/
	See http://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
	Result is a matrix where each row is a 4-tuple of [weight(count), colorchan1, colorchan2, colorchan3]
	"""

    # we could use scipy clustering and sklearn clustering..
    # i tried both for learning purposes and to compare speeds; results seem near identical
    flag_use_sklearn = False

    # report time taken
    t0 = time.time()

    color_rank = None

    if flag_use_sklearn:
        # sklearn k-means clustering
        from sklearn.cluster import KMeans
        clt = KMeans(n_clusters=numClusters)
        clt.fit(pixelList)
        # print "Sklearn clusters:"
        # centroids = clt.cluster_centers_
        # print centroids
        if (True):
            # now we need to RANK the clusters!
            # See http://www.alanzucconi.com/2015/05/24/how-to-find-the-main-colours-in-an-image/
            # Finds how many pixels are in each cluster
            hist = imgClusterColorsFromPixelList_centroid_histogram(clt, False)
            # print "histogram:"
            # print hist
            # Sort the clusters according to how many pixel they have
            color_rank = np.column_stack((hist, clt.cluster_centers_))
            # sorts by cluster weight (descending)
            color_rank = color_rank[np.argsort(color_rank[:, 0])][::-1]
        # print "Sklearn COLOR RANK:"
        # print color_rank
    else:
        # scipy k-means clustering
        from scipy.cluster.vq import vq, kmeans
        pixelListFloats = np.array(pixelList, dtype=np.float32)
        codebook, distortion = kmeans(pixelListFloats, numClusters)
        # print codebook
        if (True):
            # now we need to RANK the clusters!
            # from http://giusedroid.blogspot.com/2015/04/using-python-and-k-means-in-hsv-color.html
            # generates clusters
            data, dist = vq(pixelListFloats, codebook)
            # calculates the number of elements for each cluster
            colorCount = len(codebook)
            weights = [len(data[data == i]) for i in range(0, colorCount)]
            # creates a 4 column matrix in which the first element is the weight and the other three
            # represent the h, s and v values for each cluster
            color_rank = np.column_stack((weights, codebook))
            # sorts by cluster weight (descending)
            color_rank = color_rank[np.argsort(color_rank[:, 0])][::-1]
        # print "scipy COLOR RANK:"
        # print color_rank

    # takes the final time
    t1 = time.time()
    # print "Color clusterization took %0.5f seconds" % (t1-t0)

    return color_rank


def imgClusterColorsFromPixelList_centroid_histogram(clt, flag_dosum):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")

    if (flag_dosum):
        hist /= hist.sum()

    # return the histogram
    return hist


def imgClusterColorsFromPixelList_drawHistogramFromColorClusters(color_rank, option_clusterColorSpace):
    """
	Draw a nice histogram of colors to help visualize
	See http://giusedroid.blogspot.com/2015/04/using-python-and-k-means-in-hsv-color.html
	"""
    # creates a new blank image
    # ATTN:  uint8 could cause problem if we are drawing in HSV/or LAB color spaces?
    imagedtype = np.uint8

    # new_image =  np.array([0,0,255], dtype=imagedtype) * np.ones( (500, 500, 3), dtype=imagedtype)
    new_image = np.array([1, 1, 1], dtype=imagedtype) * np.ones((300, 300, 3), dtype=imagedtype)
    img_height = new_image.shape[0]
    img_width = new_image.shape[1]

    tot_pixels = 0
    max_pixels = 0
    for i, c in enumerate(color_rank[::-1]):
        weight = c[0]
        tot_pixels += weight
        if (weight > max_pixels):
            max_pixels = weight

    # normalize to max
    norm_pixels = max_pixels

    # for each cluster
    for i, c in enumerate(color_rank[::]):

        # gets the weight of the cluster
        weight = c[0]

        # calculates the height and width of the bins
        height = int(weight / float(norm_pixels) * img_height)
        width = img_width / len(color_rank)

        # calculates the position of the bin
        x_pos = i * width

        # defines a color so that if less than three channels have been used
        # for clustering, the color has average saturation and luminosity value
        # color = np.array( [0,128,200], dtype=imagedtype)
        color = np.array([0, 0, 0], dtype=imagedtype)

        # substitutes the known color components in the default color
        for j in range(len(c[1:])):
            color[j] = c[j + 1]

        # draws the bin to the image
        new_image[img_height - height:img_height, x_pos:x_pos + width] = [color[0], color[1], color[2]]

    # unconvert the colorspace histogram to colors of original image?
    new_image = dicerfuncs.convertColorSpaceToBgr(new_image, option_clusterColorSpace)

    # returns the cluster representation
    return new_image


# -----------------------------------------------------------











# -----------------------------------------------------------
def quantizeColorsInImage(img, color_rank, option_clusterColorSpace):
    """Quantize the colors in an image giving color_rank."""

    # create an indexImage, a 1-channel image(matrix) where each pixel value is the index number of the "closest" color
    indexImage = quantizeColorsInImageToIndexImage(img, color_rank, option_clusterColorSpace)

    # now convert indexImage to a BGR for display
    coloredImage = convertIndexImageToBgr(indexImage, color_rank, option_clusterColorSpace)

    return coloredImage


def makeClusterColorMapImageForColorSpace(color_rank, from_clusterColorSpace, to_clusterColorSpace):
    """Make some helper data objects for our color ranks, for faster operations later.
	color_rank is a matrix where each row is a 4-tuple of weight, c0,c1,c2 in color space provided by from_clusterColorSpace.
	we want to prodce a single crow "image" where each pixel is the color code for the colorrank column index in to_clusterColorSpace
	"""
    # create new single row image in stored format
    rowimage = makeClusterColorMapImage(color_rank)
    # convert from source to target
    rowimage = dicerfuncs.convertColorSpaceToColorSpace(rowimage, from_clusterColorSpace, to_clusterColorSpace)
    return rowimage


def makeClusterColorMapImage(color_rank):
    """Make nice single row image of color_ranks.
	color_rank is a matrix where each row is a 4-tuple of weight, c0,c1,c2 in color space provided by from_clusterColorSpace.
	note that this can take values from float to 8-bit ints, losing some percision
	"""
    colorCount = len(color_rank)
    rowimage = np.zeros((1, colorCount, 3), np.uint8)
    # fill it
    for i in range(0, colorCount):
        clusterColor = color_rank[i][1:]
        rowimage[0][i] = clusterColor
    return rowimage


def convertIndexImageToBgr(indexImage, color_rank, option_clusterColorSpace):
    """Convert indexImage to a BGR for display."""

    # build helper map in BGR mapping color_ranks to BGR -- this will let us look up any pixel index in color space as its BGR equiv
    bgrColorMap = makeClusterColorMapImageForColorSpace(color_rank, option_clusterColorSpace, "BGR")

    # make new target BGR image
    height, width = indexImage.shape[:2]
    colorImage = np.zeros((height, width, 3), np.uint8)

    # now walk indexImage and convert
    for y in range(0, height):
        for x in range(0, width):
            colorIndex = indexImage[y][x]
            colorImage[y][x] = bgrColorMap[0][colorIndex]

    # done, return it
    return colorImage


def convertIndexImageToBgrFiltered(indexImage, color_rank, option_clusterColorSpace, targetIndex, backgroundColor):
    """Convert indexImage to a BGR for display."""

    # build helper map in BGR mapping color_ranks to BGR -- this will let us look up any pixel index in color space as its BGR equiv
    bgrColorMap = makeClusterColorMapImageForColorSpace(color_rank, option_clusterColorSpace, "BGR")

    # make new target BGR image
    height, width = indexImage.shape[:2]
    colorImage = np.zeros((height, width, 3), np.uint8)

    # now walk indexImage and convert
    for y in range(0, height):
        for x in range(0, width):
            colorIndex = indexImage[y][x]
            if (colorIndex != targetIndex):
                colorImage[y][x] = backgroundColor
            else:
                colorImage[y][x] = bgrColorMap[0][colorIndex]

    # done, return it
    return colorImage


def quantizeColorsInImageToIndexImage(img, color_rank, option_clusterColorSpace):
    """Produce a 1-channel int matrix (image) same size as img, with 3-tuple color pixels in option_clusterColorSpace mapped to closed indexed color index in color_rank."""

    # make new target 1-chan image
    height, width = img.shape[:2]
    indexImage = np.zeros((height, width, 1), np.uint8)
    print "Image size is %dx%d" % (height, width)

    # convert bgr img to our same color_rank colorspace
    img = dicerfuncs.convertBgrToColorSpace(img, option_clusterColorSpace)

    print "Image size is %dx%d" % (height, width)

    # helper for faster lookups
    color_rank_pixels_mat = makeClusterColorMapImage(color_rank)

    # now, for speed we would like to convert these arrays to float
    if True:
        img = img.astype(float)
        color_rank_pixels_mat = color_rank_pixels_mat.astype(float)
        flag_assumeFloats = True
    else:
        flag_assumeFloats = False

    # now walk indexImage and convert
    for y in range(0, height):
        for x in range(0, width):
            pixelColor = img[y][x]
            closestColorIndex = findClosestColorIndex(pixelColor, color_rank_pixels_mat[0], option_clusterColorSpace,
                                                      flag_assumeFloats)
            indexImage[y][x] = closestColorIndex

    # done, return it
    return indexImage


def findClosestColorIndexFromColorClutsters(pixelColor, colorClusters, option_clusterColorSpace):
    """Wrapper around call."""
    # make helper matrix
    color_rank_pixels_mat = makeClusterColorMapImage(colorClusters)
    color_rank_pixels_mat = color_rank_pixels_mat.astype(float)
    flag_assumeFloats = True
    # get closest
    closestColorIndex = findClosestColorIndex(pixelColor, color_rank_pixels_mat[0], option_clusterColorSpace,
                                              flag_assumeFloats)
    return closestColorIndex


def findClosestColorIndex(pixelColor, color_rank_pixels, option_clusterColorSpace, flag_assumeFloats):
    """Find the closest color in color_rank to the specified 3-tuple pixelColor
	ideally we would use option_clusterColorSpace to be smarter about distance metrics,
	but for now assume that euclidian distance is what we want and hope caller puts us in good color space for that.
	"""
    colorCount = len(color_rank_pixels)
    closestDist = 99999999.0
    closestColorIndex = None

    for i in range(0, colorCount):
        rankColor = color_rank_pixels[i]
        if (flag_assumeFloats):
            dist = computeColorDistanceFastFloats(rankColor, pixelColor, option_clusterColorSpace)
        else:
            dist = computeColorDistanceInts(rankColor, pixelColor, option_clusterColorSpace)
        if (dist < closestDist):
            closestDist = dist
            closestColorIndex = i

    return closestColorIndex


def computeColorDistanceInts(color1, color2, option_clusterColorSpace):
    """Computer the distance between two pixel colors."""
    # ATTN: we need to make sure the np norm doesnt truncate the subtract here, so for now we do manually
    if True:
        # do we trust the math
        dist = np.linalg.norm(color1 - color2)
        return dist
    else:
        # nope
        color1f = color1.astype(float)
        color2f = color2.astype(float)
        dist = np.linalg.norm(color1f - color2f)
        return dist

    dist = math.sqrt((color1f[0] - color2f[0]) ** 2 + (color1f[1] - color2f[1]) ** 2 + (color1f[2] - color2f[2]) ** 2)
    return dist


def computeColorDistanceFastFloats(color1f, color2f, option_clusterColorSpace):
    """Computer the distance between two pixel colors."""
    dist = math.sqrt((color1f[0] - color2f[0]) ** 2 + (color1f[1] - color2f[1]) ** 2 + (color1f[2] - color2f[2]) ** 2)
    return dist


# -----------------------------------------------------------



# -----------------------------------------------------------
def mergeColorClusters(color_rank, numColorsMergeTo, option_clusterColorSpace):
    """Experimental idea of merging nearby colors...
	The idea here is that k-means clustering is heavily biased towards covering "populated" color space,
	wheras we are more concerned with covering color space BROADLY, so by k-means on a large k and then merging nearby colors we hope to ameliorate this problem
	"""

    while (len(color_rank) > numColorsMergeTo):
        # get color distance map
        colorDistanceMap = calcColorDistanceMapFromColorClusters(color_rank, option_clusterColorSpace)
        # now iteratively find and merge pair of CLOSEST colors until we are at our target count
        # note that each time we merge we probably want to adjust color_rank and recalc
        (colori, colorj) = calcClosestColors(colorDistanceMap)
        color_rank = mergeColorPairInColorRank(colori, colorj, color_rank, option_clusterColorSpace)

    # return result
    return color_rank


def calcClosestColors(colorDistanceMap):
    """Find closest pair of colors in distance map..."""

    colorCount = len(colorDistanceMap[0])
    closesti = None
    closestj = None
    closestDistance = 99999999.0
    for i in range(0, colorCount):
        for j in range(0, i):
            dist = colorDistanceMap[i][j]
            if (dist < closestDistance):
                closestDistance = dist
                closesti = i
                closestj = j
    return (i, j)


def mergeColorPairInColorRank(colori, colorj, color_rank, option_clusterColorSpace):
    """Merge two colors given by their INDICES, and adjust color_rank matrix."""
    # get COLORS of the indexed colors
    pixelcolori = color_rank[colori][1:]
    pixelcolorj = color_rank[colorj][1:]
    # calc new color for merged item
    mergedcolor = calcMergedColor(pixelcolori, pixelcolorj, option_clusterColorSpace)
    summedweight = color_rank[colori][0] + color_rank[colorj][0]
    # now replace index i with merged one
    color_rank[colori][0] = summedweight
    color_rank[colori][1] = mergedcolor[0]
    color_rank[colori][2] = mergedcolor[1]
    color_rank[colori][3] = mergedcolor[2]
    # new REMOVE colorj index
    color_rank = np.delete(color_rank, (colorj), axis=0)
    # return it
    return color_rank


def calcMergedColor(pixelcolori, pixelcolorj, option_clusterColorSpace):
    """Calculated merged (averaged) color between two 3-tuples.
	pixelcolors are FLOATS.
	"""
    sumColor = pixelcolori + pixelcolorj
    avgColor = sumColor / 2.0
    return avgColor


def calcColorDistanceMapFromColorClusters(color_rank, option_clusterColorSpace):
    """Make color-2-color distance map.
	ATTN: at a future time we might extend this to offer more complicated distance metric (for example combine/min/max/avg LAB,HSV,YCR distance metrics).
	"""

    # helper for faster lookups
    color_rank_pixels = makeClusterColorMapImage(color_rank)[0]
    color_rank_pixels = color_rank_pixels.astype(float)

    # make array
    colorCount = len(color_rank_pixels)
    colorDistanceMap = np.zeros((colorCount, colorCount, 1), np.float)

    # now we're going to make a color x color map and store distances between colors below the diagonal
    for i in range(0, colorCount):
        colori = color_rank_pixels[i]
        for j in range(0, i):
            colorj = color_rank_pixels[j]
            dist = computeColorDistanceFastFloats(colori, colorj, option_clusterColorSpace)
            colorDistanceMap[i][j] = dist

    return colorDistanceMap


# -----------------------------------------------------------



# -----------------------------------------------------------
def showQuantizedColorPlates(img, color_rank, option_clusterColorSpace):
    """Show one window for each color separation."""

    # create an indexImage, a 1-channel image(matrix) where each pixel value is the index number of the "closest" color
    indexImage = quantizeColorsInImageToIndexImage(img, color_rank, option_clusterColorSpace)

    backgroundColor = np.zeros((3), np.uint8)
    backgroundColor[0] = 254
    backgroundColor[1] = 254
    backgroundColor[2] = 0

    #  now make color plate windows
    colorCount = len(color_rank)
    for i in range(0, colorCount):
        # now convert indexImage to a BGR for display
        coloredImage = convertIndexImageToBgrFiltered(indexImage, color_rank, option_clusterColorSpace, i,
                                                      backgroundColor)
        wintitle = "Color plate %d" % i
        dicerfuncs.cvImgShow(wintitle, coloredImage)

        if (False):
            vis = coloredImage.copy()
            coloredImage = dicerfuncs.ensureGrayscale(coloredImage, False)
            mser = cv2.MSER_create()
            regions = mser.detectRegions(coloredImage, None)
            hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
            cv2.polylines(vis, hulls, 1, (0, 255, 0))
            dicerfuncs.cvImgShow(wintitle + " mser", vis)

        if (False):
            coloredImage = dicerfuncs.ensureGrayscale(coloredImage, False)
            # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
            # coloredImage = cv2.erode(coloredImage, kernel, iterations = 1)
            imgCanny = auto_canny(coloredImage)
            dicerfuncs.cvImgShow(wintitle + " canny", imgCanny)


def showQuantizedColorPlateByIndex(img, color_rank, option_clusterColorSpace, option_onlyIndex):
    """Show one window for each color separation."""

    # create an indexImage, a 1-channel image(matrix) where each pixel value is the index number of the "closest" color
    indexImage = quantizeColorsInImageToIndexImage(img, color_rank, option_clusterColorSpace)

    backgroundColor = np.zeros((3), np.uint8)
    backgroundColor[0] = 254
    backgroundColor[1] = 254
    backgroundColor[2] = 0

    coloredImage = convertIndexImageToBgrFiltered(indexImage, color_rank, option_clusterColorSpace, option_onlyIndex,
                                                  backgroundColor)
    wintitle = "Selected color plate"
    dicerfuncs.cvImgShow(wintitle, coloredImage)


def getQuantizedColorPlateByIndex(img, color_rank, option_clusterColorSpace, option_onlyIndex):
    """Show one window for each color separation."""

    # make new target 1-chan image
    height, width = img.shape[:2]

    # grayscale plate image
    imgPlate = np.zeros((height, width, 1), np.uint8)

    # convert bgr img to our same color_rank colorspace
    img = dicerfuncs.convertBgrToColorSpace(img, option_clusterColorSpace)

    # helper for faster lookups
    color_rank_pixels_mat = makeClusterColorMapImage(color_rank)
    # now, for speed we would like to convert these arrays to float
    color_rank_pixels_mat = color_rank_pixels_mat.astype(float)
    flag_assumeFloats = True

    # now walk indexImage and convert
    for y in range(0, height):
        for x in range(0, width):
            pixelColor = img[y][x]
            closestColorIndex = findClosestColorIndex(pixelColor, color_rank_pixels_mat[0], option_clusterColorSpace,
                                                      flag_assumeFloats)
            if (closestColorIndex == option_onlyIndex):
                imgPlate[y][x] = 255

    return imgPlate


# -----------------------------------------------------------




# -----------------------------------------------------------
def calcSmallestPercentageWeight(color_rank):
    """Return weight of smallest cluster as % of total."""

    colorCount = len(color_rank)
    smallestWeight = 999999.0
    sumWeight = 0.0

    for i in range(0, colorCount):
        weight = color_rank[i][0]
        sumWeight = sumWeight + weight
        if weight < smallestWeight:
            smallestWeight = weight
            smallestWeightIndex = i
    smallestWeightPercentage = (smallestWeight / sumWeight) * 100.0
    return (smallestWeightPercentage, smallestWeightIndex)


# -----------------------------------------------------------




































# -----------------------------------------------------------
def doColorPickForDie(img, dieProfile, pixelColorForeground, flag_debug):
    """User has indicated this is foreground color for die, so save it."""
    dieProfile.set_colorForeground(pixelColorForeground)
    # print "Set foreground color for die to:"
    # print pixelColorForeground

    # ok first step, find hull image
    flag_debug = True
    (img, imgMask, maskHull) = dieextract.simpleSingleForegroundObjectMaskAndCrop(img, dieProfile, flag_debug)
    dicerfuncs.cvImgShow('Color pick image', img)
    percentageCutoff = 15
    if (maskHull is None):
        return

    # now quantize colors into top N (6-8 seems reasonable)
    numColors = 4
    # convert to color format for better/different color distances (BGR, HSV, LAB, YCR)
    # my experience HSV does not do so great (see gold on red die is noisy); LAB and YCR seem pretty good; test with small numColors
    option_clusterColorSpace = "LAB"

    # convert pixelColorForeground to colorspace!
    pixelColorForeground = dicerfuncs.convertBgrToColorSpaceSinglePixel(pixelColorForeground, option_clusterColorSpace)

    # get color clusters
    colorClusters = colorQuantizeDie(img, imgMask, dieProfile, numColors, option_clusterColorSpace,
                                     pixelColorForeground, flag_debug)

    # ok now figure out what index is our pixelColorForeground
    closestColorIndex = findClosestColorIndexFromColorClutsters(pixelColorForeground, colorClusters,
                                                                option_clusterColorSpace)

    # store info in dieProfile
    dieProfile.set_quantizedColors(colorClusters, closestColorIndex, option_clusterColorSpace)

    # if we wanted we could now REPLACE closest with it
    if (True):
        colorClusters[closestColorIndex][1:] = pixelColorForeground.astype("float")

    # debug show a color plate of it
    if (flag_debug):
        imgPlate = getQuantizedColorPlateByIndex(img, colorClusters, option_clusterColorSpace, closestColorIndex)
        dicerfuncs.cvImgShow("Colorplate foreground", imgPlate)


# -----------------------------------------------------------




# -----------------------------------------------------------
def colorQuantizeDie(img, imgMask, dieProfile, numColors, option_clusterColorSpace, biasedColor, flag_debug):
    """Quantize masked image into numColors colors using k-means."""

    # color compression
    numColorsMergeTo = 999
    # this says to scan approximately X% of center rows and columns (so 50 would skip 25% from top, bottom, left, rigth, scanning an area width/2 x height/2
    # on some dice, setting this to like 50 instead of 100 really helps picking up font color with small # of colors
    scanPercent = 100
    # resize to standard small size to make color scanning faster? (0 means no target pixel size i.e. dont shrink)
    # option_pixelCountGoal = 0
    option_pixelCountGoal = 200 * 200

    # ATTN: we may be given a biasedColor.  if so this is the color we care about
    # we have a couple of options -- we could artifically seed it in the k-means
    # and/or we could do a post-facto analysis of the suggested k means and CHOOSE or REPLACE the closest one with biasedColor
    # the point is, we think we care about our biasedColor.

    # find color clusters
    colorClusters = imgFindDominantColors(img, imgMask, numColors, scanPercent, option_pixelCountGoal,
                                          option_clusterColorSpace, flag_debug)

    if (False):
        if (numColorsMergeTo < len(colorClusters)):
            # compress color clusters further with merging
            colorClusters = mergeColorClusters(colorClusters, numColorsMergeTo, option_clusterColorSpace)
            print "Reduced to %d clusters." % len(colorClusters)
            #
            if (not (colorClusters is None)):
                quantizedImage = quantizeColorsInImage(img, colorClusters, option_clusterColorSpace)
                if (flag_debug):
                    dicerfuncs.cvImgShow("Quantized colors", quantizedImage)
                    showQuantizedColorPlates(img, colorClusters, option_clusterColorSpace)

    # done
    return colorClusters
# -----------------------------------------------------------







# -----------------------------------------------------------
def imgShrinkToPixelCount(img, pixelCountGoal):
    """Return a copy of img, shrink if needed."""

    height, width = img.shape[:2]
    imgSize = height * width
    # already small enough, just return it
    if (imgSize <= pixelCountGoal):
        return dicerfuncs.copyCvImage(img)

    sizeAdjust = float(pixelCountGoal) / float(imgSize)
    newsize = float(imgSize) * sizeAdjust
    ratio = float(width) / float(height)

    # might be an explicit solution to this, but lets brute force it since i cant think of it off-hand
    for h in range(height, 1, -1):
        w = ratio * float(h)
        # we have a candidate h,w, see if its small enough
        if (h * w <= newsize):
            neww = int(w)
            newh = int(h)
            break

    # now do the resize
    # this INTER_AREA alg is recommended for shrinking
    imgSmall = cv2.resize(img, (newh, neww), interpolation=cv2.INTER_AREA)
    # print "Shrunk image from %dx%d to %dx%d" % (height,width,newh,neww)
    return imgSmall
# -----------------------------------------------------------