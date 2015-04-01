#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function

from glob import glob
from os.path import join

import kcrecog.imgproc as ip
import matplotlib.pyplot as pl
import numpy as np
from kcrecog.dataio import extract_label, extract_token

try:
    from conf import *
except ImportError:
    print("No test configuration. Please copy and adapt the template.")
    exit(-1)


def pytest_generate_tests(metafunc):
    label_files = glob(join(LABELDIR, '*.txt'))
    if 'label_file' in metafunc.funcargnames:
        metafunc.parametrize('label_file', label_files)


def _label_plot(img, labels=np.empty((0, 2)), ax=None):
    ax = ax if ax is not None else pl.gca()

    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.imshow(img)
    ax.scatter(labels[:, 0], labels[:, 1], color='r', marker='+', s=40)
    ax.axis((0, img.shape[1], img.shape[0], 0))


def _plot_rec(x0, x1, ax=None):
    """Plot rectangle with upper left x0 and lower right x1"""
    ax = ax if ax is not None else pl.gca()

    ax.plot([x0[0], x1[0], x1[0], x0[0], x0[0]],
            [x0[1], x0[1], x1[1], x1[1], x0[1]],
            ls='--', color='b')


def _get_prior_accum(label, token, nr_sample_images=10):
    fnames = (_token_to_path(label, i)
              for i in range(token - nr_sample_images, token))
    return np.mean([ip.load_valid_img(fname) for fname in fnames], axis=0,
                   dtype=int).astype(np.uint8)


def _token_to_path(label, token):
    return join(DATADIR, label + '_' + '%04d' % token + '.b16')


def _bbox_margin(x0, x1, points):
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0])
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1])

    return max(xmin - x0[0], x1[0] - xmax), max(ymin - x0[1], x1[1] - ymax)


def test_findframe_nonempty(label_file):
    """
    Test the findframe routine with the given label file by computing the
    interesting frame from the average over the NR_SAMPLES_IMAGES prior images.
    This can fail either if the returned frame is too small (does not contain
    all the labeled points) or if there are not enough valid prior images.
    """
    labels = np.loadtxt(label_file)
    if len(labels) == 0:
        return

    label, token = extract_label(label_file), int(extract_token(label_file))
    accum = _get_prior_accum(label, token)

    try:
        x0, x1 = ip.find_frame(accum)
        x0, x1 = np.array(x0), np.array(x1)
    except ip.InvalidImage as e:
        pl.figure(0, figsize=(8, 4))
        pl.title("test_findframe_nonempty: found invalid frame.")
        _label_plot(accum, ax=pl.subplot(111))
        _plot_rec(*e._debuginfo)
        pl.savefig('debugimg/' + label + '_' + '%04d' % token + '.png')
        pl.close()
        raise e

    # Check whether all the labels are within the bounding rect
    labels_in_bb = np.all(x0[None, :] < labels, axis=1) \
        * np.all(x1[None, :] > labels, axis=1)
    all_labels_in_bb = np.all(labels_in_bb)

    margin_x, margin_y = _bbox_margin(x0, x1, labels)
    bbox_to_large = (margin_x > 50) and (margin_y > 20)

    if (not all_labels_in_bb) or bbox_to_large:
        errmsg = (not all_labels_in_bb) * "Not all labels within bounding box."
        errmsg += bbox_to_large * " Bounding box too large."
        img = ip.load_valid_img(_token_to_path(label, token))
        pl.figure(0, figsize=(8, 8))
        _label_plot(img, labels, ax=pl.subplot(211))
        _label_plot(accum, labels[np.negative(labels_in_bb)],
                    ax=pl.subplot(212))
        _plot_rec(x0, x1)

        pl.subplot(211)
        pl.title("test_findframe: " + errmsg)
        pl.savefig('debugimg/' + label + '_' + '%04d' % token + '.png')
        np.save('debugimg/' + label + '_' + '%04d' % token, accum)
        pl.close()

        raise AssertionError(errmsg)


# def test_findframe_nonempty(arg1):
    # """@todo: Docstring for test_findframe_nonempty.

    # :param arg1: @todo
    # :returns: @todo

    # """

        # pl.figure(0, figsize=(8, 4))
        # pl.title("test_findframe: Found frame in empty picture")
        # _label_plot(accum, ax=pl.subplot(111))
        # _plot_rec(x0, x1)
        # pl.savefig('debugimg/' + label + '_' + '%04d' % token + '.png')
        # pl.close()
        # return

        # # raise AssertionError("Found bounding box in empty image")
