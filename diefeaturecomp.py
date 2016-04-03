# !/usr/bin/python

# dicer
# version 1.0, 1/11/16
# based heavily on code from http://www.markfickett.com/dice github code dicehistogram-master\group.py
# see also http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html

import cv2
import numpy


# -----------------------------------------------------------
def compareImageAgainstAnotherImageGetScore_Features(img1, img2, flag_debug):
  """Compare two images by using rotation-invariant feature extractors.
  Return a score consisting of the # of matching features within holography alignment tolerance."""

  # parameters
  filterMatchRatio = 0.75


  # create a detector and matcher object
  detector, matcher = createDetectorMatcher()

  # error if no descriptors were created for either image
  features1, descriptors1 = (detector.detectAndCompute(img1, None))
  if descriptors1 is None or not len(descriptors1):
    print "No features in img1: %d" % len(features1)
    return 0.0
  features2, descriptors2 = (detector.detectAndCompute(img2, None))
  if descriptors2 is None or not len(descriptors2):
    print "No features in img2: %d."  % len(features2)
    return 0.0

  # calc matches between features
  raw_matches = matcher.knnMatch(descriptors1, trainDescriptors=descriptors2, k=2)
  p1, p2, matching_feature_pairs = filterMatches(features1, features2, raw_matches, filterMatchRatio)

  # now that we have features lined up, we want to see if there is actually a nice homography transform (rotation, scale) that is consistent with bringing features into alignment.

  # numpy arrays and constants used below
  origin = numpy.array([0,0,1])
  dx = numpy.array([1,0,1])
  dy = numpy.array([0,1,1])

  # default returns
  match_count = 0
  scale_amount = float('Inf')
  
  # We need at least 4 points to align.
  if len(p1)>=4:
    homography_mat, inlier_pt_mask = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
    if homography_mat is not None:
      match_count = numpy.sum(inlier_pt_mask)
      # Sometimes matching faces are visible but the die is rotated. That is,
      # this die has 5 on top but 19 visible to the side, and the other die
      # has 19 on top but 5 visible. OpenCV may find a match, but the match
      # will not be pure translation/rotation, and will distort scale.
      h = homography_mat
      scale_amount = sum([abs(1.0 - numpy.linalg.norm(h.dot(dv) - h.dot(origin))) for dv in (dx, dy)])
      if scale_amount < 1.0:
        scale_amount = (1.0 / scale_amount if scale_amount > 0 else float('Inf'))

  # we may want to test scale_amount and disallow the matches if holography alignment scale is too far from 1.0

  return match_count
# -----------------------------------------------------------




# -----------------------------------------------------------
def filterMatches(features_a, features_b, raw_matches, ratio):
  """Returns the subset of features which match between the two lists."""
  matching_features_a, matching_features_b = [], []
  for m in raw_matches:
    if len(m) == 2 and m[0].distance < m[1].distance * ratio:
      matching_features_a.append(features_a[m[0].queryIdx])
      matching_features_b.append(features_b[m[0].trainIdx])
  p1 = numpy.float32([kp.pt for kp in matching_features_a])
  p2 = numpy.float32([kp.pt for kp in matching_features_b])
  return p1, p2, zip(matching_features_a, matching_features_b)
# -----------------------------------------------------------





# -----------------------------------------------------------
def createDetectorMatcher():
  """Create a detector and matcher for features"""
  #detector = cv2.BRISK_create()
  detector = cv2.AKAZE_create()
  matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
  return (detector, matcher)
# -----------------------------------------------------------

