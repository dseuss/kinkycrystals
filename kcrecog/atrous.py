#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function

import matplotlib.pyplot as pl
import numpy as np
from scipy.ndimage.filters import convolve
from skimage.filter import threshold_otsu

from .dataio import read_b16
from tools.helpers import Progress

# 3-spline mother kernel
MOTHER_WAVELET = [1/16, 1/4, 3/8, 1/4, 1/16]


def atrous_wavelet(source, level, mother_wavelet=MOTHER_WAVELET):
    """@todo: Docstring for atrous_wavelet.

    :param img: @todo
    :param level: @todo
    :param mother_wavelet: @todo
    :returns: [A_level, W_level, ..., W_1]

    """
    img = source.copy()
    kernel = np.array(mother_wavelet)
    wavelets = []

    for i in Progress(range(level)):
        img_smooth = _smoothen(img, kernel)
        kernel = _interleave_zeros(kernel)
        wavelet = img_smooth - img
        _threshold_to_zero(wavelet, _estimate_threshold(wavelet))
        wavelets.append(wavelet)
        img = img_smooth

    return [img] + wavelets


def _smoothen(img, kernel=MOTHER_WAVELET):
    """Convolves the image in each dimension with the 1D convolution kernel.

    :param img: 2D array containing grayscale image
    :param kernel: 1D convolution Kernel of shape (2N + 1)
    :returns: 2D array containing the convolved image in grayscale

    """
    kernel_arr = np.asarray(kernel)
    kernel_full = kernel_arr[None, :] * kernel_arr[:, None]
    return convolve(img, kernel_full, mode='reflect')


def _interleave_zeros(arr):
    """Interleaves zeros between the values of arr, i.e.

        (x_0,...,x_n) -> (x_0, 0, x_1, ..., 0, x_n)

    :param arr: Array to be interleaved with zeros
    :returns: Array interleaved with zeros

    >>> list(interleave_zeros([1, 2, 3]))
    [1, 0, 2, 0, 3]
    """
    newarr = [0 * arr[0]] * (2 * len(arr) - 1)
    newarr[::2] = arr
    return np.array(newarr)


def _threshold_to_zero(img, threshold):
    """Replaces values in img below threshold by 0 in place.

    :param img: ndarray to be thresholded
    :param threshold: Threshold

    >>> a = np.array([1, 3, 2, 4]); _threshold_to_zero(a, 2.5); list(a)
    [0, 3, 0, 4]
    """
    img[img < threshold] = 0


def _estimate_threshold(img, coeff=3):
    """Estimates the threshold used for noise supression using the MAD est.

                        t = coeff * sigma / 0.67 ,

    where sigma is the media absolute deviation of the wavelet coefficients
    from their median and the coeff is customary taken to be 3.

    :param img: Image to be thresholded
    :param coeff: Coefficient used in the thresholding formula (default: 3)
    :returns: Thresholding value t

    """
    sigma = np.median(np.abs(img - np.median(img)))
    return coeff * sigma / 0.67


if __name__ == '__main__':
    from tools.plot import imsshow
    pl.figure(0, figsize=(8, 16))

    # img = read_b16('tests/data/2015_02_13_17_14_0033.b16')
    img = read_b16('tests/data/2015_02_13_17_14_0048.b16')
    # img = read_b16('/Volumes/TOSHIBA EXT/PCO/2014_12_10_10ms_0182.b16')

    LEVELS_MIN = 4
    LEVELS_MAX = 7
    wavelets = atrous_wavelet(img, LEVELS_MAX)
    recov = [np.log(1 + np.prod(wavelets[1:i], axis=0))
             for i in range(LEVELS_MIN, LEVELS_MAX + 1)]

    axs = imsshow([img] + recov, layout='v', show=False)
    labels = ['Original'] + ['Recov. for J={}'.format(i)
                             for i in range(LEVELS_MIN, LEVELS_MAX + 1)]
    # labels += ['W_{}'.format(i) for i in range(1, len(wavelets))]
    for ax, label in zip(axs, labels):
        ax.set_title(label)

    pl.tight_layout()
    pl.show()
