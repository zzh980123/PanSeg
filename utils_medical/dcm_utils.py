import glob
import re
import sys
from typing import Union

import cv2
import numpy as np
import pydicom as dicom
from pydicom import FileDataset
from pydicom.dicomdir import DicomDir
from scipy import ndimage as ndi
# import imageio
import scipy.misc
import scipy
import os
from skimage import transform


def load_scan(path):
    slices = list()
    files = glob.glob(os.path.join(path, '**.dcm'), recursive=True)
    for file in files:
        if os.path.isfile(file):
            if not file.endswith('.dcm'):
                continue
            x = dicom.dcmread(file, force=True)
            if not hasattr(x, 'ImagePositionPatient'):
                print('missing \'ImagePositionPatient\', skip data: {}' .format(file))
                continue
            slices.append(x)

    slices.sort(key=lambda i: int(x.ImagePositionPatient[2]))

    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        if len(slices) <= 2:
            return None
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)  # SliceLocation：表示的图像平面的相对位置。

    for s in slices:
        # correct the thickness
        if s.SliceThickness <= 0:
            s.SliceThickness = slice_thickness  # 切片厚度
    return slices


def get_pixels_hu(slices, background=-2000):
    """
    To do: from network
    :param slices:
    :param background:
    :return:
    """

    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == background] = 0
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept  # Intercept
        slope = slices[slice_number].RescaleSlope  # Rescale
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16)


def get_pixels(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    return np.array(image, dtype=np.int16)


def get_meta(scan):
    spacing = map(float, ([scan[0].SliceThickness] + list(scan[0].PixelSpacing)))
    return tuple(spacing)


def resample(scans_stack, scan: Union[FileDataset, DicomDir] = None, old_spacing: tuple = None, new_spacing=(1, 1, 1),
             interpolation=0, order=3, mode='constant'):
    """
    :param scans_stack: the arrays stack at axis 0
    :param scan: the scans by load_scans
    :param new_spacing: the new space to resample
    :param interpolation: the interpolation code: 0 means use the slowest bline method, and also active the :param ord, mode
            if in range of 1-5, will use the opencv/skimage's resample interpolation:
    :param order: only enabled when interpolation is 0
    :param mode: only enabled when interpolation is 0
    :return:
    """
    assert 0 <= interpolation < 6
    assert scan is not None or old_spacing is not None

    if old_spacing is not None:
        old_spacing = np.array(list(old_spacing))

    spacing = old_spacing

    # Determine current pixel spacing
    if scan is not None:
        spacing = map(float, ([scan[0].SliceThickness] + list(scan[0].PixelSpacing)))
        spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = scans_stack.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / scans_stack.shape
    new_spacing = spacing / real_resize_factor
    if interpolation == 0:
        scans_stack = ndi.interpolation.zoom(scans_stack, real_resize_factor, order=order, mode=mode)
    elif scan is not None and len(scan) == 1 and interpolation < 5:
        scans_stack = cv2.resize(scans_stack, new_shape,
                                 interpolation=interpolation - 1)  # nearest, linear, cubic, area
    else:
        scans_stack = transform.resize(scans_stack, new_shape)
    return scans_stack, new_spacing


def load_sigle_volfile(datafile, np_var='vol', space_read=False):
    """
    this method is extract from voxelmorph#datagenerators
    load volume file
    formats: nii, nii.gz, mgz, npz, npy
    if it's a npz (compressed numpy), variable names innp_var (default: 'vol_data')
    """
    assert datafile.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file'

    if datafile.endswith(('.nii', '.nii.gz', '.mgz')):
        import nibabel as nib
        if 'nibabel' not in sys.modules:
            try:
                import nibabel as nib

            except:
                print('Failed to import nibabel. need nibabel library for these data file types.')

        X = nib.load(datafile)
        spaces = X.header.get_zooms()
        X = X.get_data()

        if space_read:
            return X, spaces

    else:  # npz, npy

        y = np.load(datafile)
        if isinstance(y, np.lib.npyio.NpzFile):
            if len(y.files) == 1:
                np_var = y.files[0]
            if np_var is None:
                np_var = 'vol'
            X = y[np_var]
            y.close()
        else:
            return y

    return X


def save_npz(path, x: np.ndarray):
    np.savez_compressed(path, vol=x)
    return 'vol'
