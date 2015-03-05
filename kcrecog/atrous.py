#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function

import matplotlib.pyplot as pl
import numpy as np
from scipy.ndimage.filters import convolve

from .dataio import read_b16
from tools.helpers import Progress

MOTHER_WAVELET = [1/16, 1/4, 3/8, 1/4, 1/16]


def smoothen(img, kernel=MOTHER_WAVELET):
    """Convolves the image in each dimension with the 1D convolution kernel.

    :param img: 2D array containing grayscale image
    :param kernel: 1D convolution Kernel of shape (2N + 1)
    :returns: 2D array containing the convolved image in grayscale

    """
    center = (len(kernel) - 1) // 2
    assert 2 * center + 1 == len(kernel), "Kernel must have uneven length"

    kernel_arr = np.asarray(kernel)
    kernel_full = kernel_arr[None, :] * kernel_arr[:, None]
    return convolve(img, kernel_full, mode='reflect')


def interleave_zeros(arr):
    """Interleaves zeros between the values of arr, i.e.

        (x_0,...,x_n) -> (x_0, 0, x_1, ..., 0, x_n)

    :param arr: Array to be interleaved with zeros
    :returns: Array interleaved with zeros

    >>> list(interleave_zeros([1, 2, 3]))
    [1, 0, 2, 0, 3]
    """
    newarr = [0 * arr[0]] * (2 * len(arr) - 1)
    newarr[::2] = arr
    return newarr


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
        img_smooth = smoothen(img, kernel)
        kernel = interleave_zeros(kernel)
        wavelets.append(img_smooth - img)
        img = img_smooth
        kernel = np.array(interleave_zeros(kernel))

    return [img] + wavelets


if __name__ == '__main__':
    from tools.plot import imsshow
    from scipy.ndimage import zoom
    pl.figure(0, figsize=(8, 16))

    # img = read_b16('tests/data/2015_02_13_17_14_0033.b16')
    img = read_b16('/Volumes/TOSHIBA EXT/PCO/2014_12_10_10ms_0182.b16')
    # img = zoom(img, .5)

    wavelets = atrous_wavelet(img, 5)
    recov = np.prod(wavelets[1:], axis=0)

    axs = imsshow([img, recov] + wavelets, layout='v', show=False)
    labels = ['Original', 'Recovery', 'Rest']
    labels += ['W_{}'.format(i) for i in range(1, len(wavelets))]
    for ax, label in zip(axs, labels):
        ax.set_title(label)

    pl.tight_layout()
    pl.show()
