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
def doClusterAutoFolder(dieProfile, subdir, flag_debug):
    """Auto cluster the images in the dieProfile 'auto' folder."""

    # params
    # Note that our aim here is to ensure we have enough images to see each side at least once, and minimize computation otherwise
    # but note that reducing the # of images used for clustering CAN result in inferior clustering (set to None or a value to multiple min needed by, e.g. 1.0 or 2.0)
    option_MinimizeFileComparisonsMultiple = 5
    option_ProbabilitySeeAll = 0.99
    option_NumClusterMultiplier = 1.5

    # base path where report will be written
    reportBasePath = dieProfile.get_reportDirectory()

    # how many clusters
    numClusters = dieProfile.get_dieFaces()
    dieFaceCount = dieProfile.get_dieFaces()

    # experimental -- cluster into more groups
    # this will greatly reduce false binning but result in more bins
    numClusters = int(numClusters * option_NumClusterMultiplier)

    # get file list (this will be the AUTO directory)
    fileList = dieProfile.get_fileList(True, subdir, dicerfuncs.get_filepattern_images())

    if (len(fileList)<numClusters):
        print "ERROR: Cannot build similarity array for unlabeled images because there are only %d images < %d clustered." % (len(fileList), numClusters)
        return None

    # how many files should we use to cluster (by default all of them)
    fileCountToCluster = len(fileList)

    # how many images do we need to see to have high likelyhood to see each side
    maxCountNeeded = diestats.rollsNeededForProbabilitySeeingAllSides(dieFaceCount, option_ProbabilitySeeAll)
    print "With a %d-sided die, to achieve probability %0.2f of seeing each side at least once, we should have at least %d rolls.  Available images: %d" % (dieFaceCount, option_ProbabilitySeeAll, maxCountNeeded, len(fileList))

    # max needed files for high probability of getting one of each
    if (not option_MinimizeFileComparisonsMultiple is None):
        #print "dieFaceCount = %d" % dieFaceCount
        #print "maxcount needed = %d" % maxCountNeeded
        fileCountToCluster = int(min( max(maxCountNeeded*option_MinimizeFileComparisonsMultiple, dieFaceCount), len(fileList) ))
        print "Using %d of %d images to do clustering (%f probability of seeing one of each side * %f)." % (fileCountToCluster, len(fileList), option_ProbabilitySeeAll, option_MinimizeFileComparisonsMultiple)


    # we will be using sklearn.cluster.AffinityPropagation to cluster
    print "Building similarity array for %d unlabeled images (%d clusters). Please wait, this could take a while.." % (fileCountToCluster, numClusters)

    # create similarity array
    time0 = time.time()
    similarityArray = buildImageComparisonSimilarityArrayFromFileList(dieProfile, fileList, fileCountToCluster, flag_debug)
    elapsedTime = time.time() - time0
    print "Built similarity array in %d seconds." % elapsedTime

    # print similarityArray

    # print "File list:"
    # print fileList

    print "Computing %d best clusters, please wait.." % numClusters

    # AGLOMMERATIVE CLUSTERING WORKS FOR US and let's us specify the target # clusters
    time0 = time.time()
    clusterer = sklearn.cluster.AgglomerativeClustering(affinity="precomputed", n_clusters=numClusters,
                                                        linkage="average")
    clusterer.fit(-1.0 * similarityArray)
    elapsedTime = time.time() - time0
    #print "Built clusters in %d seconds." % elapsedTime

    # AFFINITY PROPAGATION  does not let us directly specify # clusters, so we would have to hunt for it
    # also, it seems to be giving us odd results, at least in our tests...
    # clusterer = clusterAffinityFitToSpecificClusterCount(len(fileList), similarityArray, numClusters)


    # get cluster dictionary
    fileClusterDictionary = calcClusterDictionary(clusterer.labels_)

    # build report on files (NOTE THAT THIS CAN ONLY REPORT THE FILES WE USED TO BUILD SIMILARITY MATRIX WITH, NOT ANY WE EXCLUDED).
    htmlreport = buildHtmlReportOfClusters(reportBasePath, dieProfile, fileList, fileClusterDictionary)

    # write it out
    reportFPath = reportBasePath + "/" + "cluster_report_" + dieProfile.get_uniqueLabel() + ".html"
    dicerfuncs.writeTextToFile(htmlreport, reportFPath)
    print "Clustering report generated: %s." % reportFPath

    # now save cluster files
    saveClusterRepresentatives(dieProfile, fileList, fileClusterDictionary, "autolabeled", similarityArray, fileCountToCluster)




def buildHtmlReportOfClusters(reportBasePath, dieProfile, fileList, fileClusterDictionary):
    """Generate text html report."""
    html = ""
    html = html + "Cluster report for %s:" % dieProfile.get_uniqueLabel() + "<br/>"

    # iterate
    for key, value in fileClusterDictionary.iteritems():
        html = html + "<hr/>"
        html = html + ("Class %d:<br/>\n" % (int(key) + 1))
        for index in value:
            html = html + '<img src="%s"/> , ' % dicerfuncs.removeBaseDir(reportBasePath,fileList[index])
    # done
    return html


def calcClusterDictionary(clusterLabels):
    """Build dictionary with key -> list."""
    dict = {}
    for i, clusterNum in enumerate(clusterLabels):
        if (clusterNum in dict):
            dict[clusterNum].append(i)
        else:
            dict[clusterNum] = list()
            dict[clusterNum].append(i)
    return dict





def saveClusterRepresentatives(dieProfile, fileList, fileClusterDictionary, subdir, similarityArray, fileCountToCluster):
    """Save representatatives from each cluster to subdir, clearing contents first."""

    # params
    # this will only save prototypes/representatives needed to separate classes
    option_pruneFilePerClass = True

    # destination directory
    outdir = dieProfile.get_subdir(subdir)
    # erase any existing files
    dicerfuncs.clearContentsOfDirectory(outdir,(".png",".bmp",".jpg"))
    # now copy files
    for key, value in fileClusterDictionary.iteritems():
        excludeFileIdList = copy.deepcopy(value)
        classId = int(key)
        childCount = 0
        thisClassList = []
        for index in value:
            childCount = childCount + 1
            sourcefile = fileList[index]
            # do we need to write this file?
            if (option_pruneFilePerClass and childCount>1):
                # ideally we'd like to smartly write out children 2+ ONLY if they are needed for differentiation
                # the way to check this is to ask if this image would be classified correctly with the files we have already in this class, compared to all others
                # now we want to ask if this file would still classify to this cluster
                # note that the EFFICIENT way to do this would be to use our precomputed similarity matrix
                #print "Examining candidate File %s in cluster %s" % (sourcefile,key)
                if (False):
                    # old inefficient way
                    # build cluster dictionary but not including us
                    fileClusterDictionary_reduced = copy.deepcopy(fileClusterDictionary)
                    # remove this list for us and include only the ones we've written so far
                    fileClusterDictionary_reduced[key] = thisClassList
                    # compare files
                    clusterid = findBestScoringCluster(sourcefile, fileList, fileClusterDictionary_reduced, dieProfile)
                else:
                    # new faster way
                    clusterid = findBestScoringClusterUsingSimilarityArray(index, fileList, fileClusterDictionary, similarityArray, excludeFileIdList, fileCountToCluster)
                #
                if (clusterid == key):
                    # we don't need this file, it still maps to its original cluster
                    needThisFile = False
                else:
                    # we need this file, without it the file maps to wrong cluster
                    needThisFile = True
                #print "File %s needed: %s" % (sourcefile, str(needThisFile))
                if (not needThisFile):
                    # we dont need it
                    continue
            # save it
            classidstr = str(classId)
            if (len(classidstr)==1):
                classidstr = "0"+classidstr
            savedir = outdir + "/" + "c"+classidstr
            dicerfuncs.copyFileToDir(sourcefile, savedir)
            # remove it from exclusion list
            excludeFileIdList.remove(index)
            # add it to this class list
            thisClassList.append(index)
    # done








def findBestScoringClusterUsingSimilarityArray(fileIndex, fileList, fileClusterDictionary, similarityArray, excludeFileIdList, fileCountToCluster):
    """Using a fileClusterDictionary, find the best matching cluster id for image sourcefile."""

    bestIndex = None
    bestscore = -99999.99
    for i in range(0,fileCountToCluster):
        if i in excludeFileIdList:
            # this file not being considered (should include itself)
            continue
        score = similarityArray[fileIndex][i]
        if (score>=bestscore):
            bestscore = score
            bestIndex = i
    if (bestIndex is None):
        return None

    # ok now we have best index, now we ask what cluster that one is in
    clusterId = findClusterIdForFileIndex(bestIndex, fileList, fileClusterDictionary)
    #print "BBest match is %s with %f [CLUSTER %d]" % (fileList[bestIndex], bestscore, clusterId)
    return clusterId




def findBestScoringCluster(sourcefile, fileList, fileClusterDictionary, dieProfile):
    """Using a fileClusterDictionary, find the best matching cluster id for image sourcefile."""
    # first load the sourcefile
    flag_debug = False
    img = dicerfuncs.loadImgGeneric(sourcefile)
    img = dicerfuncs.ensureGrayscale(img, False)
    # now make list of files in fileClusterDictionary
    clusterfileList = getFileListOfClusterDictionary(fileClusterDictionary, fileList)
    # now find best match
    scoreList = compareImageAgainstFileListGetScores(dieProfile, img, clusterfileList, flag_debug)
    # sort (normalize if needed) all comparison scores so we have them in rank order
    scoreList = sortFileScores(scoreList)
    # now get best match
    (bestfpath, bestscore, bestAngle, img1a, img1r, bestImgDif, img2a, img2r, bestconfidence) = chooseBestScoringFile(scoreList, dieProfile)
    # now find the cluster id for bestfpath
    clusterId = findClusterIdForFilePath(bestfpath, fileList, fileClusterDictionary)
    #print "ABest match is %s with %f [CLUSTER %d]" % (bestfpath, bestscore, clusterId)
    return clusterId



def getFileListOfClusterDictionary(fileClusterDictionary, fileList):
    """Return file list of cluster dictionary."""
    clusterfileList = []
    for key, value in fileClusterDictionary.iteritems():
        for index in value:
            sourcefile = fileList[index]
            clusterfileList.append(sourcefile)
    return clusterfileList


def findClusterIdForFilePath(fpath, fileList, fileClusterDictionary):
    """Find which cluster a file is in by path."""
    for key, value in fileClusterDictionary.iteritems():
        for index in value:
            sourcefile = fileList[index]
            if fpath == sourcefile:
                return key
    return None


def findClusterIdForFileIndex(fileIndex, fileList, fileClusterDictionary):
    """Find which cluster a file is in by fileid."""
    for key, value in fileClusterDictionary.iteritems():
        for index in value:
            if fileIndex == index:
                return key
    return None



def buildImageComparisonSimilarityArrayFromFileList(dieProfile, fileList, fileCountToCluster, flag_debug):
    """Build an image-image similarity matrix showing how similar each file is with one another."""

    # ok first step is build a NxN matrix of scores of every pair of files

    # create similarity Array
    similarityArray = np.array(0.0, dtype=np.float32) * np.zeros((fileCountToCluster, fileCountToCluster), dtype=np.float32)

    for i in range(0, fileCountToCluster):
        fpath1 = fileList[i]
        # self similar
        similarityArray[i][i] = 1.0
        # load first image
        img1 = dicerfuncs.loadPngNoTransparency(fpath1)
        img1 = dicerfuncs.ensureGrayscale(img1, False)
        # now walk bottom half of triangle
        for j in range(i + 1, fileCountToCluster):
            fpath2 = fileList[j]
            # load second image
            img2 = dicerfuncs.loadPngNoTransparency(fpath2)
            img2 = dicerfuncs.ensureGrayscale(img2, False)
            # score
            result = diecompare.compareImageAgainstAnotherImageGetScore(dieProfile, img1, img2, flag_debug)
            score = result[0]
            # save it (and the opposite)
            similarityArray[i][j] = score
            similarityArray[j][i] = score

    return similarityArray


def guessLabelOfImageUsingFaceClusters(imgForeground, dieProfile, flag_debug):
    """Just like  guessLabelOfImage but instead of using a DIRECTORY of images, we use clusters we have computed earlier."""
    pass


# -----------------------------------------------------------





















# -----------------------------------------------------------
def clusterAffinityFitToSpecificClusterCount(sideSize, similarityArray, numClusters):
    """Use AffinityPropagation but targeting a specific number of clusters."""
    # see http://www.psi.toronto.edu/affinitypropagation/faq.html
    # see http://www.mathworks.com/matlabcentral/fileexchange/14620-cvap--cluster-validity-analysis-platform--cluster-analysis-and-validation-tool-/content/apclusterK.m

    minPref = -3.0
    maxPref = 3.0

    sameValAsLastTimeCount = 0
    clusterCountFoundLast = 0
    maxSameStop = 25

    while (True):
        pref = ((maxPref - minPref) / 2.0) + minPref
        prefv = np.array(pref, dtype=np.float32) * np.ones((sideSize), dtype=np.float32)
        clusterer = sklearn.cluster.AffinityPropagation(affinity="precomputed", preference=prefv)
        clusterer.fit(similarityArray)
        clusterCountFound = len(clusterer.cluster_centers_indices_)
        print "Trying pref %f found us %d clusters." % (pref, clusterCountFound)
        if (clusterCountFound == numClusters):
            # found it
            return clusterer
        #
        if (clusterCountFound != clusterCountFoundLast):
            # different value than last time, reset counter
            clusterCountFoundLast = clusterCountFound
            sameValAsLastTimeCount = 0
        else:
            # same value as last time, keep track how many in a row like this
            sameValAsLastTimeCount = sameValAsLastTimeCount + 1
            if (sameValAsLastTimeCount > maxSameStop):
                # stuck, we have to stop
                print "Stopping clusterAffinityFitToSpecificClusterCount hunting because it's too small a difference"
                return clusterer
        # adjust using bisection
        if (clusterCountFound < numClusters):
            minPref = pref
        else:
            maxPref = pref

# -----------------------------------------------------------

















