#!/usr/bin/env python
# encoding: utf-8
"""
Usage:
    do labeling [--datadir=<dd>] [--labeldir=<ld>] [--fileglob=<fg>]
    do view [--datadir=<dd>] [--labeldir=<ld>] [--fileglob=<fg>]

Options:
    --datadir=<dd>   Where to look for the data files [default: tests/data]
    --labeldir=<ld>  Where to look for the label files [default: tests/labels]
    --fileglob=<fg>  Globbing pattern to use [default: *.b16]
"""


from __future__ import division, print_function

import sys
import warnings
from glob import glob
from os import listdir, path
from random import shuffle

import docopt

import matplotlib as mpl
import numpy as np
from kcrecog.dataio import read_b16
from matplotlib import pyplot as pl


MOUSE_ACCURACY = 5


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


def euclidian_distance(x, y):
    """Computes the euclidian distance between x and y"""
    return np.sqrt(sum(x_i - y_i for (x_i, y_i) in zip(x, y))**2)


class LabelingDialog():
    """Figure used for labeling data by hand"""

    def __init__(self, *args, **kwargs):
        """@todo: to be defined1. """
        fig = pl.figure(0)
        pl.gray()
        fig.canvas.mpl_connect('button_press_event', self._on_mouse_click)
        fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.ax = pl.subplot(111)
        self._img = None
        self._pointlist = []

    def redraw(self):
        current_view = self.ax.axis()
        self.ax.clear()
        imshow(self._img, ax=self.ax)
        self.ax.scatter([x for x, _ in self._pointlist],
                        [y for _, y in self._pointlist],
                        color='r', marker='+', s=40)
        self.ax.axis(current_view)
        pl.draw()

    def imshow(self, img):
        self._pointlist = []
        self._img = img
        self.ax.axis((0, img.shape[1], img.shape[0], 0))
        self.redraw()

    def _on_mouse_click(self, event):
        if pl.get_current_fig_manager().toolbar.mode != '':
            return

        coord = (event.xdata, event.ydata)
        # left mouse button
        if event.button == 1:
            self._pointlist.append(coord)

        # right mouse button
        if event.button == 3:
            # FIXME This is kinda hacky, but necessary since list is changed?
            for i in reversed(range(len(self._pointlist))):
                if euclidian_distance(coord, self._pointlist[i]) < MOUSE_ACCURACY:
                    self._pointlist.pop(i)

        self.redraw()

    def _on_key_press(self, event):
        if event.key == 'ctrl+q':
            sys.exit(0)


def run_labeler(abs_fileglob, labeldir):
    print("Globbing for " + abs_fileglob + "...", end='')
    fnames = glob(abs_fileglob)
    print("{} matches found.".format(len(fnames)))
    shuffle(fnames)

    while True:
        fname = fnames.pop()
        print("Reading {}...".format(fname), end='')
        img = read_b16(fname)
        if np.max(img) == np.min(img):
            print("Invalid file.")
            continue

        fig = LabelingDialog()
        fig.imshow(img)
        pl.ioff()
        pl.show()

        newname = path.splitext(path.split(fname)[1])[0] + '.txt'
        np.savetxt(path.join(labeldir, newname),
                   np.array(fig._pointlist, dtype=float))
        print("Saved {} points".format(len(fig._pointlist)))


# def run_viewer():
    # for fname in listdir(LABELDIR):
        # datafile = path.join(DATADIR, path.splitext(path.split(fname)[1])[0] + '.b16')
        # img = read_b16(datafile)
        # with warnings.catch_warnings():
            # warnings.filterwarnings("ignore", category=UserWarning, append=1)
            # pointlist = np.loadtxt(path.join(LABELDIR, fname))

        # print("Found {} ions in {}".format(len(pointlist), fname))

        # ax = pl.gca()
        # ax.clear()
        # imshow(img)
        # ax.scatter([x for x, _ in pointlist], [y for _, y in pointlist],
                # color='r', marker='+', s=40)
        # pl.show()

if __name__ == '__main__':
    args = docopt.docopt(__doc__)

    if args['labeling']:
        run_labeler(path.join(args['--datadir'], args['--fileglob']),
                    args['--labeldir'])
    # elif args['view']:
        # run_viewer()
