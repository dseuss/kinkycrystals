#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function
import numpy as np


def from_b16(fname):
    """Reads the raw CCD camera data from a b16 file and returns as a 2D
    numpy array.

    :param fname: Path to file to read from
    :returns: numpy.ndarray[ndim=2, dtype=np.uint16]

    b16 File layout:
    ================

    Hex offset | Size | Type   | Description
    ----------------------------------------------------
    08         | 4B   | Int32  | Total header length
    0C         | 4B   | Int32  | Width of image in px
    10         | 4B   | Int32  | Height of image in px
    1C         | 4B   | Int32  | Black border?
    20         | 4B   | Int32  | White border?
    ----------------------------------------------------
    Headerlen. |      | UInt16 | Data in row major order

    """
    with open(fname, mode='rb') as infile:
        buf = np.fromfile(infile, dtype=np.int32, count=9)
        header_len = buf[2]
        imwidth = buf[3]
        imheight = buf[4]
        black_border = buf[7]
        white_border = buf[8]

        infile.seek(header_len)     # Skip the rest of the header

        data = np.fromfile(infile, dtype=np.uint16)
        print("BB={}, WB={}".format(black_border, white_border))

    return data.reshape((imheight, imwidth))


if __name__ == '__main__':
    from tools.plot import imshow
    data = from_b16('../test.b16')
    imshow(data)
