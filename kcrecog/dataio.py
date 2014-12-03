#!/usr/bin/env python
# encoding: utf-8
"""Routines for the processing of the raw data files from the CCD camera"""

from __future__ import division, print_function

from os import path

import numpy as np


def read_b16(fname):
    """Reads the raw CCD camera data from a b16 file and returns it as a 2D
    numpy array. Also applies the thresholding given by the CCD camera.

    :param str fname: Path to file to read from
    :returns: data as numpy.ndarray[ndim=2, dtype=np.uint16], with the minimal
        lower threshold at zero

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

    data[data < tmin] = tmin
    data[data > tmax] = tmax
    data -= tmin
    return data


def extract_label(fname):
    """Extracts the file label (the part of the filename between the last '_'
    and the file extension)

    :param str fname: Full path or just the file name
    :returns: Label of the file
    """
    filelabel = path.splitext(path.split(fname)[1])[0]
    return filelabel.split('_')[-1]
