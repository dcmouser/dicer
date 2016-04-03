# !/usr/bin/python

# dicer - diestats.py
# version 1.0, 2/16/16
# mouser@donationcoder.com


# -----------------------------------------------------------
# dice stats helper functions
# -----------------------------------------------------------

# imports
import dicerfuncs
#from dicerfuncs import *

import os
import math
import operator as op

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







def buildHtmlReportStatisticsPairwise(reportBasePath, dieProfile, pairwiseCounts):
    htmlReport = ''

    # a pretty pariwise heatmap
    htmlReport += "<br/><hr/>Heatmap for Previous vs Subsequent Rolls:<br/>"
    htmlReportStats = buildHtmlReportHeatmap(reportBasePath, dieProfile, pairwiseCounts)
    htmlReport += htmlReportStats

    # separate stats/plot for each prior roll
    pkeys = pairwiseCounts.keys()
    pkeys.sort()
    for pLabel in pkeys:
        imgsubname = dieProfile.get_uniqueLabel()+"x"+pLabel
        htmlReport += "<br/><hr/>Pairwise stats for subsequent roll when prior roll is %s:<br/>" % pLabel
        htmlReportStats = buildHtmlReportStatistics(reportBasePath, dieProfile, pairwiseCounts[pLabel], imgsubname)
        htmlReport += htmlReportStats
    return htmlReport





def buildHtmlReportOfFileLabels(reportBasePath, dieProfile, unlabeledFileList, labeledFileList, unlabeledResults, labeledClassResults, labeledFilesById):
    """Generate text html report."""
    html = ""
    html = html + "File label report for %s:" % dieProfile.get_uniqueLabel() + "<br/>"

    # iterate
    #for key, value in labeledClassResults.iteritems():
    for key in sorted(labeledClassResults):
        value = labeledClassResults[key]
        html = html + "<hr/>"
        html = html + ("Class %s:<br/>\n" % key)
        #html = html + " [" + '<img src="%s"/> , ' % labeledFilesById[key] + "]<br/>"
        for index in value:
            html = html + '<img src="%s"/> , ' % dicerfuncs.removeBaseDir(reportBasePath,unlabeledFileList[index])
    # done
    return html
# -----------------------------------------------------------




# -----------------------------------------------------------
def buildHtmlReportStatistics(reportBasePath, dieProfile, classCounts, imgsubname):
    html = ""

    # pvalue for our power test to report unfairness
    option_unfairPvalue = 0.05


    # TEST to make results seem stronger
    option_FakeStrongCountMultiply = None
    if (not option_FakeStrongCountMultiply is None):
        print "WARNING: Testing by making results seem stronger by multiplying counts by %d" % option_FakeStrongCountMultiply
        for key,val in classCounts.iteritems():
            classCounts[key] = classCounts[key]*option_FakeStrongCountMultiply

    #
    countTotals = calcTotals(classCounts)
    dieFaces = dieProfile.get_dieFaces()


    # sorted class labels and values
    (classLabels,values) = getSortedClassLabelsAndValuesFromCounts(classCounts)

    html += "Statistics report after analyzing %d rolls of this %d-sided die:<br/>" % (countTotals, dieFaces)
    html += "Frequencies: %s<br/>\n" % str(values)

    # ask dice about probabilities of each face (usually uniform)
    expectedProbabilities = dieProfile.get_expectedFaceProbabilities()
    expectedFrequencies = [x * float(countTotals) for x in expectedProbabilities]

    #print "counttotals: %d" % countTotals
    #print "expected prob:"
    #print expectedProbabilities
    #print "expected freq:"
    #print expectedFrequencies

    # chisquared statistic
    try:
        (chi2, p) = scipy.stats.chisquare( values, f_exp = expectedFrequencies)
    except Exception as e:
        estr = "ERROR RUNNING CHISQUARED: %s<br/>\n" % str(e)
        print estr
        html += estr
        (chi2, p) = (0.0,1.0)

    html += "Chisquared: %f (p=%f)<br/>\n" % (chi2, p)
    html += "Likelyhood of a fair die yielding results as extreme as these: %2.04f%%<br/>\n" % (p*100.0)

    # calc some hint about power
    if (p<option_unfairPvalue):
        html += "Therefore, die can be confidentently labeled as unfair."
    else:
        powerUntil = calcPowerRepeatUntilUnlikely(values, option_unfairPvalue)
        if (powerUntil is None):
            html += "Power test inconclusive.<br/>\n"
        else:
            html += "Power test suggests these results would have to be repeated identically %d more times (%d more identical rolls) before die could be confidently labeled as unfair.<br/>\n" % (powerUntil, powerUntil*countTotals)

    # ok now build histogram
    img = buildFrequencyHistogramForFaces(countTotals, classCounts, expectedProbabilities, expectedFrequencies)

    # save img
    if (img is not None):
        imgfpath = reportBasePath + "/" + "filelabel_report_" + imgsubname + "_histogram.png"
        dicerfuncs.saveImgGeneric(imgfpath, img)

        # overlay expectedFrequencies with confidence intervals
        html += "<br/>Frequency histogram:<br/>\n"
        html += '<img src="%s"/>' % dicerfuncs.removeBaseDir(reportBasePath,imgfpath)

    return html



def calcTotals(classCounts):
    sum = 0
    for key,val in classCounts.iteritems():
        sum = sum + val
    return sum



def calcPowerRepeatUntilUnlikely(values, targetP):
    """Figure out how many more times these results would have to be repeated until we were confident die was unfair"""
    maxval = 9999
    valuelen = len(values)
    for i in range(2,maxval):
        newvalues = [x * i for x in values]
        #newvalues = copy.copy(values)
        #for j in range(0,valuelen):
        #    newvalues[j] = newvalues[j]*i
        (chi2, p) = scipy.stats.chisquare(newvalues)
        if (p<targetP):
            return (i-1)
    return None
# -----------------------------------------------------------











# -----------------------------------------------------------
def buildFrequencyHistogramForFaces(countTotals, classCounts, expectedProbabilities, expectedFrequencies):
    """Build a histogram frequency plot."""
    # see http://people.duke.edu/~ccc14/pcfb/numpympl/MatplotlibBarPlots.html

    # options
    option_ciprobability = 0.95
    option_showObservationErrorBars = True

    if (countTotals==0):
        return None

    # get sorted classlabels and values
    (classLabels, values) = getSortedClassLabelsAndValuesFromCounts(classCounts)

    # generate confidence intervals
    confidenceIntervals = []
    bins = len(expectedProbabilities)

    # two ways to generate this data, i still find it confusing..
    if (True):
        # see http://stackoverflow.com/questions/28242593/correct-way-to-obtain-confidence-interval-with-scipy
        variances = [(x*(1.0-x)*countTotals) for x in expectedProbabilities]
        for i in range(0,bins):
            mean = expectedFrequencies[i]
            scale = math.sqrt(variances[i])
            ci = scipy.stats.norm.interval(option_ciprobability, loc=mean, scale=scale)
            cidist = (ci[1]-ci[0])/2.0
            cival = cidist
            confidenceIntervals.append(cival)
    else:
        # stddev of pure probabilities
        # see http://math.stackexchange.com/questions/185184/statistical-significance-dice-probability
        stddev = [math.sqrt(x*(1.0-x)/countTotals) for x in expectedProbabilities]
        for i in range(0,bins):
            mean = expectedProbabilities[i]
            scale = stddev[i]
            ci = scipy.stats.norm.interval(option_ciprobability, loc=mean, scale=scale)
            cidist = (ci[1]-ci[0])/2.0
            # scale confidencer interval in probability to frequencies
            cival = cidist * countTotals
            confidenceIntervals.append(cival)


    # alternate confidence intervals not of fair die but of observed vals
    bconfidenceIntervals = []
    # see http://stackoverflow.com/questions/28242593/correct-way-to-obtain-confidence-interval-with-scipy
    bovservationProportions = [float(x)/countTotals for x in values]
    bvariances = [(x*(1.0-x)*countTotals) for x in bovservationProportions]
    for i in range(0,bins):
        mean = values[i]
        scale = math.sqrt(bvariances[i])
        ci = scipy.stats.norm.interval(option_ciprobability, loc=mean, scale=scale)
        cidist = (ci[1]-ci[0])/2.0
        # alternate
        cidist = max(mean-ci[0], ci[1]-mean)
        #
        cival = cidist
        bconfidenceIntervals.append(cival)



    #print "Confidence intervals:"
    #print confidenceIntervals


    # plot options
    binCount = len(values)
    ind = np.arange(binCount)         # the x locations for the groups
    width = .85                      # the width of the bars
    #calculate nice max height for graph
    maxes1 = [sum(x) for x in zip(confidenceIntervals,expectedFrequencies)]
    maxes = [max(x) for x in zip(values,maxes1)]
    maxes = [x*1.10 for x in maxes]
    maxYCount = max(maxes)
    # graph visual options
    option_fontsize = 9
    colorFrequencies = 'cyan'
    colorErrorBars = 'red'
    colorErrorBars2 = 'green'
    lineWidthErrorBars = 1
    errorBarCapSize = 4

    #
    ticklocations = ind+(width/2.0)


    # not sure about this
    fig, ax = plt.subplots(sharex = True)

    # plot frequencies
    rects1 = ax.bar(ind, values, width,
                    color=colorFrequencies
                    )

    # error bars
    ebar = ax.errorbar(ticklocations, expectedFrequencies, yerr=confidenceIntervals, fmt='o', color=colorErrorBars,  elinewidth=lineWidthErrorBars, capsize=errorBarCapSize)

    # error bars on bars
    if (option_showObservationErrorBars):
        ticklocations2 = ind+(width/4.0)
        ebar2 = ax.errorbar(ticklocations2, values, yerr=bconfidenceIntervals, fmt='o', color=colorErrorBars2,  elinewidth=lineWidthErrorBars, capsize=errorBarCapSize)


    # axes and labels
    ax.set_xlim(-width,len(ind)+width)
    ax.set_ylim(0,maxYCount)
    ax.set_ylabel('Frequency Count')
    ax.set_title('Die Face Observations (red is %0.2f confidence int. for fair die)' % option_ciprobability)
    xTickMarks = classLabels
    ax.set_xticks(ticklocations)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=45, fontsize=option_fontsize)

    ## add a legend
    #ax.legend( (rects1[0], ebar[0]), ('Die Rolls','fair die 95%') )


    # generate it
    # ridiculous how hard this is to find http://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
    fig.tight_layout(pad=2)
    fig.canvas.draw()
    #
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #
    img = dicerfuncs.convertRGBtoBGR(data)

    # close figure
    plt.close(fig)
    #plt.close(ax)

    # return it
    return img
# -----------------------------------------------------------


# -----------------------------------------------------------
def getSortedClassLabelsAndValuesFromCounts(classCounts):
    keys = classCounts.keys()
    keys.sort()
    classLabels = []
    values = []
    for key in keys:
        classLabels.append(key)
        values.append(classCounts[key])

    return (classLabels, values)
# -----------------------------------------------------------



# -----------------------------------------------------------
def buildHtmlReportHeatmap(reportBasePath, dieProfile, pairwiseCounts):
    """Nice heatmap."""

    html = ""

    # build img
    img = buildPairwiseHeatmap(dieProfile, pairwiseCounts)

    # save img
    imgsubname = dieProfile.get_uniqueLabel()+"_pairheatmap"
    imgfpath = reportBasePath + "/" + "filelabel_report_" + imgsubname + "_histogram.png"
    dicerfuncs.saveImgGeneric(imgfpath, img)

    # show image
    html += '<img src="%s"/>' % dicerfuncs.removeBaseDir(reportBasePath,imgfpath)

    return html




def buildPairwiseHeatmap(dieProfile, pairwiseCounts):
    """Build heatmap image."""

    # plot options
    binCount = len(pairwiseCounts)
    ind = np.arange(binCount)         # the x locations for the groups

    # plot options
    option_fontsize = 9
    width = 1.0


    # data
    data = np.zeros((binCount, binCount), np.uint)
    pkeys = pairwiseCounts.keys()
    pkeys.sort()
    for i,key in enumerate(pkeys):
        jkeys = pairwiseCounts[key].keys()
        jkeys.sort()
        for j,jkey in enumerate(jkeys):
            data[i,j]=pairwiseCounts[key][jkey]

    #
    fig, ax = plt.subplots(sharex = True)

    # plot heatmap
    #plt.pcolor(data, cmap=plt.cm.Reds)
    heatmap = plt.pcolor(data, cmap='jet', shading ='faceted')

    # axes and labels
    #ax.set_xlim(-width,len(ind)+width)
    #ax.set_ylim(-width,len(ind)+width)
    ax.set_ylabel('Previous roll')
    ax.set_xlabel('Subsequent roll')
    ax.set_title('Previous vs Subsequent Roll Counts')

    ticklocations = ind+(width/2.0)
    xTickMarks = pkeys

    ax.set_xticks(ticklocations)
    ax.set_yticks(ticklocations)
    xtickNames = ax.set_xticklabels(xTickMarks)
    ytickNames = ax.set_yticklabels(xTickMarks)
    plt.setp(xtickNames, fontsize=option_fontsize)
    plt.setp(ytickNames, fontsize=option_fontsize)


    # show labels and colormap
    # see http://stackoverflow.com/questions/11917547/how-to-annotate-heatmap-with-text-in-matplotlib
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            plt.text(x + 0.5, y + 0.5, '%d' % data[y, x],
                     horizontalalignment='center',
                     verticalalignment='center',
                     )
    plt.colorbar(heatmap)


    ## add a legend
    #ax.legend( (rects1[0], ebar[0]), ('Die Rolls','fair die 95%') )

    # generate it
    # ridiculous how hard this is to find http://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
    fig.tight_layout(pad=2)
    fig.canvas.draw()
    #
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = dicerfuncs.convertRGBtoBGR(data)

    # close figure
    plt.close(fig)
    #plt.close(ax)
    #plt.close(heatmap)

    # return it
    return img

# -----------------------------------------------------------












































# -----------------------------------------------------------
# see https://rosettacode.org/wiki/Evaluate_binomial_coefficients#Python
def comb_version1(n,r):
    """Return binomial of n choose r."""
    if r > n-r:  # for smaller intermediate values
        r = n-r
    return int( reduce( op.mul, range((n-r+1), n+1), 1) / reduce( op.mul, range(1,r+1), 1) )


# http://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python
def comb(n, r):
    """Return binomial of n choose r."""
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer//denom
# -----------------------------------------------------------




# -----------------------------------------------------------
# see http://math.stackexchange.com/questions/266505/a-number-of-dice-rolls-to-see-every-number-at-least-once?rq=1
def probSeeingAllSides(n, r):
    """Calculate probability of seeing all sides of an n-sided die at least once after r rolls."""
    sum = 0.0
    for k in range(0,n+1):
        val = (-1.0) ** k
        val = val * comb(n,k)
        kdivn = float(k)/float(n)
        val = val * math.pow( (1.0 - kdivn) , r)
        sum = sum + val
    return sum


def rollsNeededForProbabilitySeeingAllSides(n, p):
    """Calculate the number of rolls needed of an n-sided die to achieve a p probability of seeing all sides at least once."""
    # we could use bifurcation search but we'd like to avoid computing anything higher than what we need
    maxrolls = 1000
    for r in range(n, maxrolls):
        testp = probSeeingAllSides(n,r)
        #print "Test p (%f) for  probSeeingAllSides(%d,%d) is %f" % (p, n,r,testp)
        if (testp>=p):
            return r
    return None
# -----------------------------------------------------------



# -----------------------------------------------------------
def testDieStats():
    """Test functions."""
    print "testDieStats:"
    #print "comb(9,4):"
    #print comb(9,4)
    print "probSeeingAllSides(2,3):"
    print probSeeingAllSides(2,3)
    print "probSeeingAllSides(10,30):"
    print probSeeingAllSides(10,30)
    print "rollsNeededForProbabilitySeeingAllSides(10, .95):"
    print rollsNeededForProbabilitySeeingAllSides(10, .95)
    print "rollsNeededForProbabilitySeeingAllSides(20, .95):"
    print rollsNeededForProbabilitySeeingAllSides(20, .95)
# -----------------------------------------------------------




# -----------------------------------------------------------
# invoked if script is run as standalne
def main():
    """Main function"""
    testDieStats()



# invoked if script is run as standalne
if __name__ == "__main__":
    main()
# -----------------------------------------------------------