#!/usr/bin/env python
# encoding: utf-8
"""Routines for extracting data from the images."""
from __future__ import division, print_function

import cv2 as cv
import numpy as np
from itertools import izip
from skimage.measure import label, regionprops

import conf
import dataio as io

try:
    from tools.helpers import Progress
except ImportError:
    def Progress(iter):
        return iter


###############################################################################
#                      Loading sequences & frame finding                      #
###############################################################################

def read_seq(fnames):
    """Reads a sequence of images from the files given. In this sequence the
    crystal is expected to be in the same region for all pictures. This is
    used to extract to the bounding box.

    :param list fnames: List of filenames to load
    :returns: imgs, props
        imgs: Dict of 2D arrays with same shape containing the images with
            the key given by the label of the file (the part of the filename
            between the last '_' and the file extension)
        props: Dict with additional properties of the sequence
            - x0: upper left corner of the relevant region in the original
            image.
            - max: Maximum brightness value over the whole sequence

    """
    assert(len(fnames) > 0)
    imgs = [_load_img(fname) for fname in Progress(fnames)]

    # Compute "mean" image and scale to fit datatype
    sumimg = sum(img.astype(np.float) for img in imgs)
    sumimg -= sumimg.min()
    sumimg = (255 / sumimg.max() * sumimg).astype(np.uint8)

    x0, x1 = find_frame(sumimg)

    imgs = dict(izip((io.extract_label(fname) for fname in fnames),
                     (img[x0[1]:x1[1], x0[0]:x1[0]] for img in imgs)))
    props = {'x0': x0, 'max': max(np.max(img) for img in imgs.itervalues())}
    return imgs, props


def find_frame(data):
    """Finds the interesting frame of an image.

    :param data: Image as 2D array of type uint8
    :returns: (x0, x1), where x0 is the upper left corner and x1 the lower
        right corner of the bounding rectangle

    """
    assert data.dtype == np.uint8, "Image has wrong dtype."

    # TODO Separate preprocessing routine for less noisy images
    buf = _extract_interesting_region(data)
    bbox = regionprops(buf)[0].bbox
    x0, x1 = _postprocess_bb(x0=(bbox[1], bbox[0]),
                             x1=(bbox[3], bbox[2]),
                             imgsize=buf.shape)
    return x0, x1


def load_valid_img(fname):
    """Loads the image from the given b16 file and checks whether the image
    contains some information (i.e. is not unicolor).

    :param fname: Path to file to read from
    :returns: Image as 2D array in uint8 format

    """
    data = io.read_b16(fname)
    max_val, min_val = data.max(), data.min()
    if (max_val > 255) or (min_val < 0):
        raise InvalidImage("Image data range too large for uint8 format.")
    if (max_val == min_val):
        raise InvalidImage("Image is blank.")
    return np.array(data, dtype=np.uint8)


def _load_img(fname):
    """Reads the data from the file and converts it to the dataformat used.

    :param fname: Path to file to read from
    :returns: Image as 2D array in uint8 format

    """
    data = io.read_b16(fname)
    if (data.max() > 255) or (data.min() < 0):
        raise InvalidImage("Image data range too large for uint8 format.")
    return np.array(data, dtype=np.uint8)


def _extract_interesting_region(data):
    """Preprocesses image for frame finding. Steps taken are
        * Otsu thresholding
        * small-kernel-area opening to get rid of single/isolated bright pixels
          which are assumend to be noise
        * large-kernel-area opening to inverse the effect of the prior opening;
          this also serves the purpose to connect close bright areas (for the
          next step). Due to the distinct elongation of the kernel in the x-
          direction this especially favors horizontal structures.
        * finding largest connected region

    :param np.ndarray data: Image as 2D array of type uint8
    :returns: Mask as 2D array labeling the "interesting area" in that picture

    """
    assert data.dtype == np.uint8, "Image has wrong dtype."
    _, buf = cv.threshold(data, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Opening to get rid of noise
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    buf = cv.morphologyEx(buf, cv.MORPH_OPEN, kernel, iterations=3)

    # Closing to get connected area where signal should be
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (31, 7))
    buf = cv.morphologyEx(buf, cv.MORPH_CLOSE, kernel, iterations=5)

    # Find the largest connected component
    cc = label(buf, neighbors=4, background=0) + 1
    largest_component = np.argmax(np.bincount(cc.ravel())[1:]) + 1
    return cc == largest_component


def _postprocess_bb(x0, x1, imgsize):
    """Performs checking and scaling of the bounding box.

    :param x0: Upper left corner of the bounding box
    :param x1: Lower right corner of the bounding box
    :param imgisze: Maximum values for the position of the lower right corner
    :returns: New values for x0 and x1

    """
    xlen, ylen = x1[0] - x0[0], x1[1] - x0[1]
    # Check whether the bbox is to small
    if (xlen < conf.BBOX_MIN_X) or (ylen < conf.BBOX_MIN_Y):
        msg = "Bounding box too small: {} should be at least {}" \
            .format((xlen, ylen),
                    (conf.BBOX_MIN_X, conf.BBOX_MIN_Y))
        raise InvalidImage(msg, debuginfo=(x0, x1))

    # Scale the rectangle
    x0 = (max(x0[0] - xlen * (conf.BBOX_SCALE_X - 1) / 2, 0),
          max(x0[1] - ylen * (conf.BBOX_SCALE_Y - 1) / 2, 0))
    x1 = (min(x1[0] + xlen * (conf.BBOX_SCALE_X - 1) / 2, imgsize[1]),
          min(x1[1] + ylen * (conf.BBOX_SCALE_Y - 1) / 2, imgsize[0]))

    return np.floor(x0).astype(int), np.ceil(x1).astype(int)


###############################################################################
#                               Ion recognition                               #
###############################################################################

###############################################################################
#                                 Exceptions                                  #
###############################################################################

class InvalidImage(Exception):
    """Exception class to be raised when image file fails preprocessing"""

    def __init__(self, value, debuginfo=None):
        self._value = value
        self._debuginfo = debuginfo

    def __str__(self):
        return repr(self._value)
