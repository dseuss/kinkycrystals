#!/usr/bin/env python
# encoding: utf-8
"""Little tool to label images"""

from __future__ import division, print_function

from glob import glob
from matplotlib import pyplot as pl
from os import path, listdir
from random import shuffle

import numpy as np

from kcrecog.dataio import read_b16


def extract_sequence_labels(ddir):
    """Returns all unqiue sequence labels from the datadir. These are the
    parts of the file names in `ddir` that occur before the last '_'.

    :param str ddir: Path to datadir as string
    :returns: Set with sequence labels

    """
    return {'_'.join(fname.split('_')[:-1]) for fname in listdir(ddir)}


def imshow(img, ax=None, **kwargs):
    """Shows the image `img` passed as numpy array in a much prettier way

    :param np.ndarray img: Image to show passed as RGB or grayscale image
    :param ax: Axis to use for plot (default: current axis)

    """
    if ax is None:
        ax = pl.gca()

    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.imshow(img, **kwargs)
    ax.axis((0, img.shape[1], img.shape[0], 0))


def mouse_click(event, pointlist, img):
    if pl.get_current_fig_manager().toolbar.mode != '':
        return

    ax = pl.gca()
    current_axis = ax.axis()

    # left mouse button
    if event.button == 1:
        pointlist.append((event.xdata, event.ydata))

    # right mouse button
    if event.button == 3:
        for i in reversed(range(len(pointlist))):
            if (abs(event.xdata - pointlist[i][0]) < 5) \
                    and (abs(event.ydata - pointlist[i][1]) < 5):
                pointlist.pop(i)

    ax.clear()
    imshow(img)
    ax.scatter([x for x, _ in pointlist], [y for _, y in pointlist],
               color='r', marker='+', s=40)
    ax.axis(current_axis)
    pl.draw()


if __name__ == '__main__':
    DATADIR = '/data/PCO/'
    LABELDIR = 'tmp/'
    MIN_SLEN = 100

    seqlabels = {slab for slab in extract_sequence_labels(DATADIR)
                 if (len(glob(path.join(DATADIR, slab + '*.b16'))) > MIN_SLEN)
                 and (slab.startswith('2014'))}

    fnames = sum((glob(path.join(DATADIR, slab + '*.b16'))
                  for slab in seqlabels), [])
    shuffle(fnames)

    while True:
        fig = pl.figure(0)
        pl.gray()
        pointlist = []
        fname = fnames.pop()
        img = read_b16(fname)
        imshow(img)
        cid = fig.canvas.mpl_connect('button_press_event',
                                     lambda event: mouse_click(event, pointlist, img))
        pl.show()

        newname = path.join(LABELDIR, path.splitext(path.split(fname)[1])[0] + '.txt')
        np.savetxt(newname, pointlist)
        print("Saved {} points to {}".format(len(pointlist), newname))

