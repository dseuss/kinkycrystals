#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function

import cv2 as cv
import numpy as np

import conf

try:
    from tools.helpers import Progress
except ImportError:
    Progress = lambda x: x


###############################################################################
#                              Main IO Functions                              #
###############################################################################

def read_raw_data(fname):
    """Reads the raw CCD camera data from a b16 file and returns it as a 2D
    numpy array. Also applies the thresholding of the CCD camera.

    :param str fname: Path to file to read from
    :returns: data as numpy.ndarray[ndim=2, dtype=np.uint8],

    b16 File layout:
    ================

    Hex offset | Size | Type   | Description
    ----------------------------------------------------
    08         | 4B   | Int32  | Total header length
    0C         | 4B   | Int32  | Width of image in px
    10         | 4B   | Int32  | Height of image in px
    1C         | 4B   | Int32  | Lower threshold
    20         | 4B   | Int32  | Upper threshold
    ----------------------------------------------------
    Headerlen. |      | UInt16 | Data in row major order

    """
    with open(fname, mode='rb') as infile:
        buf = np.fromfile(infile, dtype=np.int32, count=9)
        header_len = buf[2]
        shape = (buf[4], buf[3])
        tmin, tmax = buf[7], buf[8]
        infile.seek(header_len)     # Skip the rest of the header

        data = np.fromfile(infile, dtype=np.uint16).reshape(shape)

    # Validate the image
    if data.max() == data.min():
        raise InvalidImage("Image seems to be empty")
    if (tmax > 255) or (tmin < 0):
        raise InvalidImage("Image range too large for uint8")

    data[data < tmin] = tmin
    data[data > tmax] = tmax
    return data.astype(np.uint8)


def read_img(fname):
    """Reads the data from the file given and finds the relevant region.

    :param fname: Path to file to read from
    :returns: Preprocessed interesting region from the image as array

    """
    data = read_raw_data(fname)
    x0, x1 = _find_frame(data, method='noisy')
    return data[x0[1]:x1[1], x0[0]:x1[0]]


def read_seq(fnames):
    """Reads a sequence of images from the files given. In this sequence the
    crystal is expected to be in the same region for all pictures. This is
    used to extract to the bounding box.

    :param list fnames: List of filenames to load
    :returns: List of 2D arrays with same shape containing the images

    """
    if len(fnames) < conf.MIN_SEQ_LEN:
        raise InvalidSequence("Sequence length {} too short, should be {}"
                              .format(len(fnames), conf.MIN_SEQ_LEN))
    imgs = []
    for fname in Progress(fnames):
        try:
            imgs.append(read_raw_data(fname))
        except InvalidImage as e:
            print(e)

    if len(fnames) < conf.MIN_SEQ_LEN:
        raise InvalidSequence("Sequence length {} too short, should be {}"
                              .format(len(fnames), conf.MIN_SEQ_LEN))

    # Compute "mean" image and scale to fit datatype
    sumimg = sum(img.astype(np.float) for img in imgs)
    sumimg -= sumimg.min()
    sumimg = (255 / sumimg.max() * sumimg).astype(np.uint8)

    x0, x1 = _find_frame(sumimg, method='clean')
    return np.array([img[x0[1]:x1[1], x0[0]:x1[0]].copy() for img in imgs],
                    dtype=np.uint8)


###############################################################################
#                          Post Processing Functions                          #
###############################################################################

def _preprocess_noisy(data):
    """Preprocesses image for noisy frame finding. Steps taken are
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
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (21, 7))
    buf = cv.morphologyEx(buf, cv.MORPH_CLOSE, kernel, iterations=5)

    # Find the largest connected component
    _, cc = cv.connectedComponents(buf.astype(np.uint8))
    largest_component = np.argmax(np.bincount(cc.ravel())[1:]) + 1
    return (cc == largest_component).astype(np.uint8)


def _find_frame(data, method='clean'):
    """Finds the interesting frame of an image.

    :param data: Image as 2D array of type uint8
    :param method: Either 'clean' or 'noisy' indicating whether a noise-
        reducing preprocessing should be used prior to frame-finding. This
        may lead to more robust results for noisy images, but accuracy may be
        lost (default: 'clean')
    :returns: (x0, x1), where x0 is the upper left corner and x1 the lower
        right corner of the bounding rectangle

    """
    assert data.dtype == np.uint8, "Image has wrong dtype."

    if method == 'clean':
        # FIXME There must be a better routine
        buf = _preprocess_noisy(data)
    elif method == 'noisy':
        buf = _preprocess_noisy(data)
    else:
        raise AttributeError("Invalid preprocessing method: " + str(method))

    rect = cv.boundingRect(buf)
    x0, x1 = _post_process_bb(x0=(rect[0], rect[1]),
                              x1=(rect[0] + rect[2], rect[1] + rect[3]),
                              imgsize=buf.shape)
    return x0, x1


def _post_process_bb(x0, x1, imgsize):
    """Performs checking and scaling of the bounding box.

    :param x0: Upper left corner of the bounding box
    :param x1: Lower right corner of the bounding box
    :param imgisze: Maximum values for the position of the lower right corner
    :returns: New values for x0 and x1

    """
    xlen, ylen = x1[0] - x0[0], x1[1] - x0[1]
    # Check whether the bbox is to small
    if (xlen < conf.BBOX_MIN_X) or (ylen < conf.BBOX_MIN_Y):
        raise InvalidImage("Bounding box too small: {} should be at least {}"
                           .format((xlen, ylen),
                                   (conf.BBOX_MIN_X, conf.BBOX_MIN_Y)))

    # Scale the rectangle
    x0 = (max(x0[0] - xlen * (conf.BBOX_SCALE_X - 1) / 2, 0),
          max(x0[1] - ylen * (conf.BBOX_SCALE_Y - 1) / 2, 0))
    x1 = (min(x1[0] + xlen * (conf.BBOX_SCALE_X - 1) / 2, imgsize[1]),
          min(x1[1] + ylen * (conf.BBOX_SCALE_Y - 1) / 2, imgsize[0]))

    return np.floor(x0).astype(int), np.ceil(x1).astype(int)


###############################################################################
#                                 Exceptions                                  #
###############################################################################

class InvalidImage(Exception):
    """Exception class to be raised when image file fails preprocessing"""

    def __init__(self, value):
        self._value = value

    def __str__(self):
        return repr(self._value)


class InvalidSequence(Exception):
    """Exception class to be raised when a sequence fails loading"""

    def __init__(self, value):
        self._value = value

    def __str__(self):
        return repr(self._value)


if __name__ == '__main__':
    from matplotlib import pyplot as pl
    from tools.plot import imshow, imsshow
    from glob import glob
    import os
    pl.gray()

    DATADIR = '/media/dsuess/TOSHIBA/PCO/'
    # DATADIR = '../data/'
    seq_labels = {'_'.join(fname.split('_')[:-1])
                  for fname in os.listdir(DATADIR)
                  if fname.startswith('2014') and fname.endswith('.b16')}

    def plot_rect(x0, x1, ax):
        x = [x0[0], x1[0], x1[0], x0[0], x0[0]]
        y = [x0[1], x0[1], x1[1], x1[1], x0[1]]
        ax.plot(x, y, color='r')

    # for fname in glob(DATADIR + '2014*.b16'):
        # imshow(read_img(fname))

    for slab in seq_labels:
        try:
            imgs = read_seq(glob(DATADIR + slab + '*.b16'))
            imsshow(imgs[:10], layout='v')
        except InvalidSequence as e:
            print(e)

    # sequence_names =
    # imgs = read_seq(glob('2014_11_06_17_49_*.b16'))
    # print(len(imgs))
    # plot_rect(imgs[0], imgs[1], pl.gca())
    # pl.show()
    # # for img in imgs:
    # #     imshow(img)
