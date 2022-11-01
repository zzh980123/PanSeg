import math

import numpy as np

from sklearn import metrics


# input should be format of CxWxHxL or CxWxH
def dice(g: np.ndarray, label: np.ndarray):
    """

    :param g: generate img format of numpy array
    :param label: the label
    :return:
    """
    if len(g.shape) < 3:
        g = g[np.newaxis, ...]
    if len(label.shape) < 3:
        label = label[np.newaxis, ...]

    if len(g.shape) != len(label.shape):
        raise RuntimeError('the lengths of the inputs must be equal')

    dims = tuple([x for x in range(1, len(g.shape))])
    num = g * label
    num_c = np.sum(num, axis=dims)
    den1 = g * g
    den1_c = np.sum(den1, axis=dims)
    den2 = label * label
    den2_c = np.sum(den2, axis=dims)

    dice_all_channel = 2 * ((num_c + 0.0000001) / (den1_c + den2_c + 0.0000001))
    channel_size = dice_all_channel.shape[0]

    dice_total = np.sum(dice_all_channel) / channel_size

    return dice_total


def dice_labels(vol1, vol2, labels=None, nargout=1):
    '''
    Dice [1] volume overlap metric

    The default is to *not* return a measure for the background layer (label = 0)

    [1] Dice, Lee R. "Measures of the amount of ecologic association between species."
    Ecology 26.3 (1945): 297-302.

    Parameters
    ----------
    vol1 : nd array. The first volume (e.g. predicted volume)
    vol2 : nd array. The second volume (e.g. "true" volume)
    labels : optional vector of labels on which to compute Dice.
        If this is not provided, Dice is computed on all non-background (non-0) labels
    nargout : optional control of output arguments. if 1, output Dice measure(s).
        if 2, output tuple of (Dice, labels)

    Output
    ------
    if nargout == 1 : dice : vector of dice measures for each labels
    if nargout == 2 : (dice, labels) : where labels is a vector of the labels on which
        dice was computed
    '''
    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))
        labels = np.delete(labels, np.where(labels == 0))  # remove background

    # import ipdb; ipdb.set_trace()
    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        top = 2 * np.sum(np.logical_and(vol1 == lab, vol2 == lab))
        bottom = np.sum(vol1 == lab) + np.sum(vol2 == lab)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom

    if nargout == 1:
        return dicem
    else:
        return (dicem, labels)


from surface_distance.metrics import *


def asd_labels(vol1, vol2, spaces=(1, 1, 1), labels=None, nargout=1):
    """
    Average Surface Distance

    Parameters
    ----------
    vol1 : nd array. The first volume (e.g. predicted volume)
    vol2 : nd array. The second volume (e.g. "true" volume)
    labels : optional vector of labels on which to compute Dice.
        If this is not provided, ASD is computed on all non-background (non-0) labels
    nargout : optional control of output arguments. if 1, output Dice measure(s).
        if 2, output tuple of (ASD, labels)

    Output
    ------
    if nargout == 1 : dice : vector of dice measures for each labels
    if nargout == 2 : (dice, labels) : where labels is a vector of the labels on which
        dice was computed
    """
    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))

    labels = np.delete(np.array(labels), np.where(labels == 0))  # remove background

    asdm = np.zeros((len(labels), 2))
    for idx, lab in enumerate(labels):
        distances = compute_surface_distances(vol2 == lab, vol1 == lab, spacing_mm=spaces)
        ads = compute_average_surface_distance(distances)
        asdm[idx][0] = ads[0]
        asdm[idx][1] = ads[1]

    if nargout == 1:
        return asdm[..., 0], asdm[..., 1]
    else:
        return (asdm, labels)


def hausdorff_labels(vol1, vol2, spaces=(1, 1, 1), percent=95, labels=None, nargout=1):
    """
    Average Surface Distance

    Parameters
    ----------
    vol1 : nd array. The first volume (e.g. predicted volume)
    vol2 : nd array. The second volume (e.g. "true" volume)
    labels : optional vector of labels on which to compute Dice.
        If this is not provided, ASD is computed on all non-background (non-0) labels
    nargout : optional control of output arguments. if 1, output Dice measure(s).
        if 2, output tuple of (ASD, labels)

    Output
    ------
    if nargout == 1 : dice : vector of dice measures for each labels
    if nargout == 2 : (dice, labels) : where labels is a vector of the labels on which
        dice was computed
    """
    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))

    labels = np.delete(np.array(labels), np.where(labels == 0))  # remove background

    hausdorffm = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        distances = compute_surface_distances(vol2 == lab, vol1 == lab, spacing_mm=spaces)
        ads = compute_robust_hausdorff(distances, percent=percent)
        hausdorffm[idx] = ads

    if nargout == 1:
        return hausdorffm
    else:
        return (hausdorffm, labels)


def RMSE(g: np.ndarray, label: np.ndarray):
    error = g - label
    mse = np.average(error * error)
    return math.sqrt(mse)


def ncc(a: np.ndarray, b: np.ndarray):
    a = (a - a.mean())
    b = (b - b.mean())
    c = np.multiply(a, b)
    c = c / (a.std() * b.std())
    return c


def MI(labels_true: np.ndarray, labels_pre: np.ndarray):
    labels_true = labels_true.flatten()
    labels_pre = labels_pre.flatten()
    return metrics.mutual_info_score(labels_true, labels_pre)


def NMI(labels_true: np.ndarray, labels_pre: np.ndarray):
    labels_true = labels_true.flatten()
    labels_pre = labels_pre.flatten()
    return metrics.normalized_mutual_info_score(labels_true, labels_pre)
