# !/usr/bin/python

# dicer
# version 1.0, 1/11/16
# (c) mouser@donationcoder.com 1/5/16 - 1/26/16

"""
Commandline Use:
  -h, --help            show this help message and exit

"""

# -----------------------------------------------------------
import dicerfuncs
# from dicerfuncs import *
from dieprofile import DieProfile
from dicecamera import DiceCamera
import diecontrol
import dieextract
import diecluster
import diecolor
#
# for argument parsing:
#import optparse
import argparse
# for open cv
import numpy as np
import cv2
# -----------------------------------------------------------



















# -----------------------------------------------------------
# main commandline actions

def do_testBasics(dieProfile):
    """Run ongoing live basic test."""
    flag_debug = True
    flag_smartChanges = False
    dicerfuncs.captureAndProcessLoop("Testing simple single object cropping", testProcess_simpleSingleObjectCrop,
                                     dieProfile, flag_debug, flag_smartChanges)


def do_testCamera(dieProfile):
    """Run ongoing pur camera display."""
    flag_debug = True
    flag_smartChanges = False
    dicerfuncs.captureAndProcessLoop("Testing camera", testProcess_Camera,
                                     dieProfile, flag_debug, flag_smartChanges)


def do_testForeground(dieProfile):
    """Run ongoing live basic test."""
    flag_debug = True
    flag_smartChanges = False
    dicerfuncs.captureAndProcessLoop("Testing foreground extraction", testProcess_simpleForegroundExtraction,
                                     dieProfile, flag_debug, flag_smartChanges)


# captureAndProcessLoop("Testing simple single object cropping", testProcess_simpleTest, flag_debug = True, flag_smartChanges=True)


def do_saveBackground(dieProfile):
    """Save background image from camera into known file location."""
    dicerfuncs.saveBackground(dieProfile)


def do_testColorQuantization(dieProfile):
    """Run ongoing color quantizationc test."""
    flag_debug = True
    flag_smartChanges = False
    dicerfuncs.captureAndProcessLoop("Testing simple single object cropping", testProcess_colorQuantization, dieProfile,
                                     flag_debug, flag_smartChanges)


def do_labelSave(dieProfile):
    """Let user save labeled images."""
    flag_debug = True
    flag_smartChanges = False
    imagedir = "labeled"
    dicerfuncs.saveCopyOfBackground(dieProfile, imagedir)
    dieProfile.useBackgroundFileIfPresentInSubdir(imagedir)
    dicerfuncs.captureAndProcessLoop(
        "Labeling and saving images for die '%s' (hit 1-9,A-P to save that die face)" % dieProfile.get_uniqueLabel(),
        doProcess_labelSave, dieProfile, flag_debug, flag_smartChanges)


def do_labelGuess(dieProfile):
    """try to guess labels of images."""
    flag_debug = True
    flag_smartChanges = True
    dicerfuncs.captureAndProcessLoop("Gussing image labels for die '%s'" % dieProfile.get_uniqueLabel(),
                                     doProcess_labelGuess, dieProfile, flag_debug, flag_smartChanges)


def do_autoSave(dieProfile):
    """Automatically capture and save die images for future clustering."""
    flag_debug = True
    flag_smartChanges = True
    imagedir = "unlabeled"
    dicerfuncs.saveCopyOfBackground(dieProfile, imagedir)
    #dieProfile.useBackgroundFileIfPresentInSubdir(imagedir)
    dicerfuncs.captureAndProcessLoop("Automatically taking periodic captures and saving in 'auto' subfolder for die '%s'" % dieProfile.get_uniqueLabel(), doProcess_autoSave, dieProfile, flag_debug, flag_smartChanges)


def do_extractForeground(dieProfile):
    """Automatically extract foregrounds for previously captured images."""
    flag_debug = True
    flag_smartChanges = True
    imagedir = "unlabeled"
    dieProfile.useBackgroundFileIfPresentInSubdir(imagedir)
    dicerfuncs.doExtractForegrounds(dieProfile, imagedir, flag_debug)



def do_autoCluster(dieProfile):
    """Analyze previously autosaved images and cluter based on # of diefaces."""
    flag_debug = False
    flag_smartChanges = True
    imagedir = "unlabeled"
    dieProfile.useBackgroundFileIfPresentInSubdir(imagedir)
    diecluster.doClusterAutoFolder(dieProfile,imagedir, flag_debug)


def do_fileLabel(dieProfile):
    """Analyze previously autosaved images and cluter based on # of diefaces."""
    flag_debug = False
    flag_smartChanges = True
    imagedir = "unlabeled"
    dieProfile.useBackgroundFileIfPresentInSubdir(imagedir)
    dicerfuncs.doFileLabel(dieProfile,imagedir, 'labeled', flag_debug)


def do_fileLabel_debug(dieProfile):
    """Analyze previously autosaved images and cluter based on # of diefaces."""
    flag_debug = True
    flag_smartChanges = True
    imagedir = "labeled"
    dieProfile.useBackgroundFileIfPresentInSubdir(imagedir)
    dicerfuncs.doFileLabel(dieProfile,imagedir, 'labeled', flag_debug)



# dicerfuncs.captureAndProcessLoop("Building clusters from images in 'auto' subfolder for die '%s'" % dieProfile.get_uniqueLabel(), doProcess_autoCluster, dieProfile, flag_debug, flag_smartChanges)

# -----------------------------------------------------------






# -----------------------------------------------------------
# helper test wrappers called from above

def testProcess_Camera(img, dieProfile, k, flag_debug):
    """Just test camera"""
    dicerfuncs.cvImgShow('Camera raw', img)
    return True


def testProcess_simpleSingleObjectCrop(img, dieProfile, k, flag_debug):
    """Try to crop out single simple foreground image"""
    (img, imgMask, maskHull) = dieextract.simpleSingleForegroundObjectMaskAndCrop(img, dieProfile, flag_debug)
    if (not maskHull is None):
        dicerfuncs.testCandidateImage(img, imgMask, maskHull, dieProfile)
    return True



def testProcess_simpleForegroundExtraction(img, dieProfile, k, flag_debug):
    """Try to crop out single simple foreground image"""
    (img, imgMask, maskHull) = dieextract.simpleSingleForegroundObjectMaskAndCrop(img, dieProfile, flag_debug)
    imgForeground = dieextract.extractForegroundFromImage(img, imgMask, maskHull, dieProfile, False, flag_debug)
    return True



def testProcess_colorQuantization(img, dieProfile, k, flag_debug):
    """Try some color quantization tests."""
    (img, imgMask, maskHull) = dieextract.simpleSingleForegroundObjectMaskAndCrop(img, dieProfile, flag_debug)
    dicerfuncs.cvImgShow('Cropped image', img)
    percentageCutoff = 15
    if (not maskHull is None):
        diecolor.testColorQuantization(img, imgMask, maskHull, dieProfile, flag_debug)
    # testColorQuantizationUnderPercent(img, imgMask, percentageCutoff, flag_debug)
    return True


def testProcess_simpleTest(img, dieProfile, k, flag_debug):
    """Try to crop out single simple foreground image"""
    (img, imgMask, maskHull) = dieextract.simpleSingleForegroundObjectMaskAndCrop(img, dieProfile, flag_debug)
    centerPercent = 50
    if (not maskHull is None):
        dicerfuncs.testCandidateImage(img, imgMask, maskHull, dieProfile)
    # img = cropToCenterRectangle(img, centerPercent)
    # imgMask = cropToCenterRectangle(imgMask, centerPercent)
    # img = dicerfuncs.imgInvert(img)
    # testCandidateImage(img, imgMask, maskHull, dieProfile)

    return True


# -----------------------------------------------------------



















# -----------------------------------------------------------
def doProcess_labelSave(img, dieProfile, k, flag_debug):
    """Let user save labeled images."""

    # show cam
    # cvImgShow('Live camera', img)

    # get foreground
    flag_isolateCenter = True
    imgForeground = dieextract.isolateAndExtractForegroundFromImage(img, dieProfile, flag_isolateCenter, flag_debug)
    if (imgForeground is None):
        return True

    # let user save it
    dicerfuncs.checkKeypressSaveImage(k, imgForeground, img, dieProfile)

    return True


def doProcess_labelGuess(img, dieProfile, k, flag_debug):
    """Guess labels of images."""

    # show cam
    # cvImgShow('Live camera', img)

    # debug early stuff?
    flag_debug1 = flag_debug and True

    # get foreground
    flag_isolateCenter = True
    imgForeground = dieextract.isolateAndExtractForegroundFromImage(img, dieProfile, flag_isolateCenter, flag_debug1)
    if (imgForeground is None):
        return True

    # guess label
    flag_debug2 = flag_debug and False
    dicerfuncs.guessLabelOfImage(imgForeground, dieProfile, 'labeled', flag_debug2)

    return True


# -----------------------------------------------------------












# -----------------------------------------------------------
def doProcess_autoSave(img, dieProfile, k, flag_debug):
    """Auto save images.
    What we do here is watch for STABLE img for some period, then save, then wait for image to CHANGE.
    """

    # debug early stuff?
    flag_debug1 = flag_debug and True

    # get foreground
    flag_isolateCenter = True
    imgForeground = dieextract.isolateAndExtractForegroundFromImage(img, dieProfile, flag_isolateCenter, flag_debug1)
    if (imgForeground is None):
        return True


    # we count on caller to only invoke us when it's time to save and image has changed
    imagedir = dieProfile.get_faceImageDir('unlabeled')
    dicerfuncs.doSaveDieFaceImage(dieProfile, imgForeground, img, imagedir)


    # auto spin die with hardware if appropriate?
    dieProfile.doAutoRollIfAppropriate()

    return True


def doProcess_autoCluster(img, dieProfile, k, flag_debug):
    """Auto save images."""

    # show cam
    # cvImgShow('Live camera', img)

    # debug early stuff?
    flag_debug1 = flag_debug and False

    # get foreground
    flag_isolateCenter = True
    imgForeground = dieextract.isolateAndExtractForegroundFromImage(img, dieProfile, flag_isolateCenter, flag_debug1)
    if (imgForeground is None):
        return True

    # guess label given clustering
    # ATTN: unfinished
    diecluster.guessLabelOfImageUsingFaceClusters(imgForeground, dieProfile, flag_debug)

    return True
# -----------------------------------------------------------


























# -----------------------------------------------------------
# invoked if script is run as standalne
def main():
    """Main function"""

    # show a welcome
    dicerfuncs.sayHello();

    # get commandline args

    # using argparse
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    #parser = optparse.OptionParser()

    # parser.add_option('--input', help='filename (optionally with path) of csv file to read', default='data.csv')
    parser.add_argument('--background', help='save background image', action="store_true", dest="flag_savebackground", default=False)
    parser.add_argument('--test', help='perform an onging live test', action="store_true", dest="flag_test", default=False)
    parser.add_argument('--testf', help='test foreground extraction', action="store_true", dest="flag_test_foreground", default=False)
    parser.add_argument('--testcam', help='test camera', action="store_true", dest="flag_test_cam", default=False)

    parser.add_argument('--testcolorq', help='do some color tests', action="store_true", dest="flag_testcolorq")
    parser.add_argument('--savelabel', help='do some labeling and comparing', action="store_true",
                      dest="flag_savelabel", default=False)
    parser.add_argument('--guesslabel', help='guess label', action="store_true", dest="flag_guesslabel",
                      default=False)
    parser.add_argument('--colorpick', help='show zoomed color picking window', action='store_true',
                      dest="flag_colorpick", default=False)
    parser.add_argument('--autosave', help='automatically capture and save die images for future clustering',
                      action='store_true', dest="flag_autosave", default=False)

    parser.add_argument('--autocluster', help='analyze previously autosaved images and cluter based on # of diefaces', action='store_true', dest="flag_autocluster", default=False)
    parser.add_argument('--filelabel', help='analyze previously autosaved images and label using labeled/ folder', action='store_true', dest="flag_filelabel", default=False)
    parser.add_argument('--filelabel_debug', help='analyze previously autosaved images and label using labeled/ folder in debug mode', action='store_true', dest="flag_filelabel_debug", default=False)
    parser.add_argument('--extractfore', help='extract foreground of perviously saved images in unlabeled/ folder', action='store_true', dest="flag_extractfore", default=False)


    # die profile options
    parser.add_argument('--dielabel', help='the unique label for a die being analyzed', action="store", dest="option_dieLabel", default=None)
    parser.add_argument('--dieshape', help='the shape of a die being analyzed', action="store", dest="option_dieShape", default=None)
    parser.add_argument('--diefaces', help='the number of unique die faces', action="store", dest="option_dieFaces", default=None)
    parser.add_argument('--diefull', help='use the full die image instead of isolating foreground?', action="store_true", dest="option_diefull", default=False)
    parser.add_argument('--diecenter', help='use die center instead of isolating foreground?', action="store_true", dest="option_diecenter", default=False)

    parser.add_argument('--cropwidth', help='crop the camera view to center width', action="store", dest="option_cropWidth", default=0)
    parser.add_argument('--cropheight', help='crop the camera view to center height', action="store", dest="option_cropHeight", default=0)
    parser.add_argument('--cropcirclex', help='crop the camera view to circle', action="store", dest="option_cropCircleX", default=0)
    parser.add_argument('--cropcircley', help='crop the camera view to circle', action="store", dest="option_cropCircleY", default=0)
    parser.add_argument('--cropcircler', help='crop the camera view to circle', action="store", dest="option_cropCircleR", default=0)

    parser.add_argument('--denoise', help='denoise extracted foreground', action="store_true", dest="option_denoise", default=False)


    parser.add_argument('--autoroll', help='auto roll dice', action="store_true", dest="option_autoroll", default=False)


    # parse options
    args = parser.parse_args()

    # get args/options
    flag_savebackground = args.flag_savebackground
    flag_test = args.flag_test
    flag_test_foreground = args.flag_test_foreground
    flag_testcolorq = args.flag_testcolorq
    flag_test_cam = args.flag_test_cam

    flag_savelabel = args.flag_savelabel
    flag_guesslabel = args.flag_guesslabel
    flag_colorpick = args.flag_colorpick
    flag_autosave = args.flag_autosave
    flag_autocluster = args.flag_autocluster
    flag_filelabel = args.flag_filelabel
    flag_filelabel_debug = args.flag_filelabel_debug
    flag_extractfore = args.flag_extractfore

    # die profile options
    option_dieLabel = args.option_dieLabel
    option_dieShape = args.option_dieShape
    option_dieFaces = args.option_dieFaces
    option_diefull = args.option_diefull
    option_diecenter = args.option_diecenter

    # other options
    option_cropWidth = args.option_cropWidth
    option_cropHeight = args.option_cropHeight
    option_cropCircleX = args.option_cropCircleX
    option_cropCircleY = args.option_cropCircleY
    option_cropCircleR = args.option_cropCircleR

    option_denoise = args.option_denoise

    # hardware
    option_autoroll = args.option_autoroll


    # die profile object (default has no information)
    dieProfile = DieProfile()
    dieProfile.set_uniqueLabel(option_dieLabel)
    dieProfile.set_dieShape(option_dieShape)
    dieProfile.set_colorPick(flag_colorpick)
    if (not option_dieFaces is None):
        dieProfile.set_dieFaces(int(option_dieFaces))

    dieProfile.set_dieFull(option_diefull)
    dieProfile.set_diecenter(option_diecenter)
    dieProfile.set_dieThreshold(option_diefull)

    #
    dieProfile.set_cropWidth(int(option_cropWidth))
    dieProfile.set_cropHeight(int(option_cropHeight))
    dieProfile.set_cropCircle(float(option_cropCircleX),float(option_cropCircleY),float(option_cropCircleR) )
    #
    dieProfile.set_autoRoll(option_autoroll)
    dieProfile.set_denoise(option_denoise)


    # process commandline
    if (flag_test_cam):
        do_testCamera(dieProfile)
    if (flag_savebackground):
        do_saveBackground(dieProfile)
    if (flag_test):
        do_testBasics(dieProfile)
    if (flag_test_foreground):
        do_testForeground(dieProfile)
    if (flag_testcolorq):
        do_testColorQuantization(dieProfile)
    if (flag_savelabel):
        do_labelSave(dieProfile)
    if (flag_guesslabel):
        do_labelGuess(dieProfile)
    #
    if (flag_autosave):
        do_autoSave(dieProfile)
    if (flag_autocluster):
        do_autoCluster(dieProfile)
    if (flag_filelabel):
        do_fileLabel(dieProfile)
    if (flag_filelabel_debug):
        do_fileLabel_debug(dieProfile)


    if (flag_extractfore):
        do_extractForeground(dieProfile)

    # end
    dicerfuncs.debugprint("dicer.py ends.")


# -----------------------------------------------------------







# -----------------------------------------------------------
# invoked if script is run as standalne
if __name__ == "__main__":
    main()
# -----------------------------------------------------------
