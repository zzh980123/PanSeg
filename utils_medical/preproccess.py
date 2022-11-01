import glob
import itertools
import math
import os
import re
from operator import itemgetter
from typing import Union, Optional, List, Tuple, AnyStr, Dict, Any

import PIL
import numpy as np
from PIL import Image
from matplotlib import figure
from pydicom import dataset, dicomdir
from pydicom import dcmread
from pydicom.pixel_data_handlers import util
from typing.io import BinaryIO

from . import dcm_utils as du, plot, module_util, numpy_util

# from . import dcm_utils as du
# import plot, module_util, numpy_util

#
# pydicom can only handle Pixel Data that hasn't been compressed, these libs( pylibjpeg pylibjpeg-libjpeg[option] ) can read
# compressed data, install them just using command: pip install; Google to get more information. By nowandfuture.
#


'''

It is a Helper script for Registration and any other medical image pre-process or some CV task.
Created by nowandfuture.

'''

try:
    import SimpleITK as sitk

    sitk_exits = True
except:
    sitk_exits = False
    print('SimpleITK is not imported.')

try:
    import ants

    ants_exits = True
except:
    ants_exits = False
    print('ants is not imported.')

BACKEND = ['imageio', 'cv2']
# bmp,jpg,png,tif,gif,pcx,tga,exif,fpx,svg,psd,cdr,pcd,dxf,ufo,eps,ai,raw,WMF,webp == replaced by (,|\[)([a-zA-Z]*) $1'$2'
# to add more, you can check wiki and opencv lib.
SUFFIX = ['bmp', 'jpg', 'png', 'tif', 'gif', 'pcx', 'tga', 'exif', 'fpx', 'svg', 'psd', 'cdr', 'pcd', 'dxf', 'ufo',
          'eps', 'ai', 'raw',
          'WMF', 'webp']


class ImageBackend(object):

    def __init__(self, module_=None):
        self.m = module_

    def read_image(self, fd, format_,
                   **kwargs) -> np.ndarray:
        raise NotImplementedError()

    def write_image(self, fd, image, format_,
                    **kwargs):
        raise NotImplementedError()


class imageio_(ImageBackend):

    def read_image(self, fd, format_=None, **kwargs):
        return self.m.imread(fd)

    def write_image(self, fd, image, format_,
                    **kwargs):
        self.m.imwrite(fd, image, format_, **kwargs)


class cv2_(ImageBackend):

    def read_image(self, fd, format_=None, **kwargs):
        return self.m.imread(fd, format_)

    def write_image(self, fd, image, format_,
                    **kwargs):
        # todo no test
        self.m.imwrite(fd, image, format_)


def __load_loader(loader_list):
    def check_bk(it):
        return it is not None

    _img_tool = None
    for b in loader_list:
        try:
            m = module_util.get_module(__name__)
            module = module_util.import_module(b)
            _img_tool = module_util.create_instance(b + '_', m, module)
            if check_bk(_img_tool):
                print("Use {} as backend".format(b))
                break
        except:
            pass

    if not check_bk(_img_tool):
        raise RuntimeWarning("No imager Loader loaded, it will cause error when reading IMAGES.")

    return _img_tool


__image_tool: ImageBackend = __load_loader(BACKEND)


def load_image_loader(module_name):
    return __load_loader([module_name]) is not None


def set_image_loader(image_loader: ImageBackend):
    __image_tool = image_loader


def read_image(fd: str, format_=None, **kwargs):
    image = __image_tool.read_image(fd, format_=format_, **kwargs)
    return np.array(image)


def read_images_as_voxel(fd_list: List[str], format_=None, **kwargs):
    slices = [read_image(file_path, format=format_
                         , **kwargs) for file_path in fd_list]
    return np.stack(slices)


def read_one_voxel(fd: str, var_data='vol', read_spaces=False):
    """
        this method is extract from voxelmorph#datagenerators
        load volume file
        formats: nii, nii.gz, mgz, npz, npy
        if it's a npz (compressed numpy), variable names innp_var (default: 'vol_data')
    """
    return du.load_sigle_volfile(fd, var_data, space_read=read_spaces)


def read_npz_voxels(fd: str, keys: Tuple):
    rt = []
    for k in keys:
        rt.append(du.load_sigle_volfile(fd, k))

    return tuple(rt)


def read_dcms_ants(dir_name: str, pixel_type='float'):
    return ants.dicom_read(dir_name, pixeltype=pixel_type)


def read_dcms_dicom(dir_name):
    slices = du.load_scan(dir_name)
    if slices is not None:
        np_voxel = du.get_pixels(slices)
        return np_voxel, du.get_meta(slices)
    return None, None


def read_dcm_dicom(fp: Union[str, "os.PathLike[AnyStr]", BinaryIO],
                   defer_size: Optional[Union[str, int]] = None,
                   stop_before_pixels: bool = False,
                   force: bool = False,
                   specific_tags: Optional[List[Union[int, str, Tuple[int]]]] = None):
    return du.dicom.dcmread(fp, defer_size, stop_before_pixels, force, specific_tags)


def write2image(array: np.ndarray, file_name: str, format_param=None, auto_create=True, **kwargs):
    shape = array.shape

    assert len(shape) <= 3
    if len(shape) == 3:
        assert 3 >= shape[2] >= 1

    check_path(os.path.split(file_name)[0], auto_create=auto_create)
    __image_tool.write_image(file_name, array, format_=format_param, **kwargs)


def write2npz(array: np.ndarray, file_name: str, auto_create=True):
    check_path(os.path.split(file_name)[0], auto_create=auto_create)
    np.savez_compressed(file_name, vol=array)
    return 'vol'


def multi_write2npz(file_name: str, auto_create=True, *args, **kwargs):
    check_path(os.path.split(file_name)[0], auto_create=auto_create)
    np.savez_compressed(file_name, *args, **kwargs)


def write2npy(array: np.ndarray, file_name, auto_create=True):
    check_path(os.path.split(file_name)[0], auto_create=auto_create)
    np.save(file_name, array)


import nibabel as nib


def write2nii(array: np.ndarray, file_name, auto_create=True):
    check_path(os.path.split(file_name)[0], auto_create=auto_create)
    new_image = nib.Nifti1Image(array, np.eye(4))
    nib.save(new_image, file_name)


def write2dcm(array: np.ndarray, file_name, auto_create=True):
    check_path(os.path.split(file_name)[0], auto_create=auto_create)

    du.dicom.dcmwrite(file_name, array)


def check_path(path: str, auto_create=True):
    if not os.path.exists(path):
        if auto_create and path != '':
            os.makedirs(path)
        return True

    return False


import json


def thru_plane_position(dcm):
    """Gets spatial coordinate of image origin whose axis
            is perpendicular to image plane.
            """
    orientation = tuple((float(o) for o in dcm.ImageOrientationPatient))
    position = tuple((float(p) for p in dcm.ImagePositionPatient))
    rowvec, colvec = orientation[:3], orientation[3:]
    normal_vector = np.cross(rowvec, colvec)
    slice_pos = np.dot(position, normal_vector)
    return slice_pos


def read_subject(dicom_dir, image_type='MR'):
    """ Read in the directory of a single subject and return a numpy array """
    directory = os.path.join(dicom_dir)

    if os.path.isdir(directory):
        files = [os.path.join(directory, fname) for fname in os.listdir(directory) if fname.endswith('.dcm')]
    else:
        files = [directory]

    # Read slices as a list before sorting
    dcm_slices = [dcmread(fname) for fname in files]

    # Extract position for each slice to sort and calculate slice spacing
    dcm_slices = [(dcm, thru_plane_position(dcm)) for dcm in dcm_slices]
    dcm_slices = sorted(dcm_slices, key=itemgetter(1))
    spacings = np.diff([dcm_slice[1] for dcm_slice in dcm_slices])
    slice_spacing = np.mean(spacings)

    # All slices will have the same in-plane shape
    shape = (int(dcm_slices[0][0].Columns), int(dcm_slices[0][0].Rows))
    nslices = len(dcm_slices)

    if image_type == 'CT':
        # Final 3D array will be N_Slices x Columns x Rows
        shape = (nslices, *shape)
        img = np.empty(shape, dtype='float32')
        for idx, (dcm, _) in enumerate(dcm_slices):
            # Rescale and shift in order to get accurate pixel values
            slope = float(dcm.RescaleSlope)
            intercept = float(dcm.RescaleIntercept)
            img[idx, ...] = dcm.pixel_array.astype('float32') * slope + intercept
    elif image_type == 'MR':
        shape = (nslices, *shape)
        img = np.empty(shape, dtype='float32')
        for idx, (dcm, _) in enumerate(dcm_slices):
            rescaled_arr = util.apply_modality_lut(dcm.pixel_array, dcm)
            img[idx, ...] = util.apply_voi_lut(rescaled_arr, dcm, index=0)
            img[idx, ...] = dcm.pixel_array.astype('float32')

    else:
        raise RuntimeError("Unknown Image type, only process CT or MRI")

    # Calculate size of a voxel in mm
    pixel_spacing = tuple(float(spac) for spac in dcm_slices[0][0].PixelSpacing)
    voxel_spacing = (slice_spacing, *pixel_spacing)

    return img, voxel_spacing


def convert2npz(dicom_dir, npz_dir, black_list: tuple = (), white_list: list = None):
    """ Converts all subjects in DICOM_DIR to 3D numpy arrays """

    subjects = []
    for root, dirs, files in os.walk(dicom_dir):
        for d in dirs:
            dname = os.path.join(root, d)
            sub = os.listdir(dname)
            if len(sub) > 0:
                for file in sub:
                    if file.endswith('.dcm') and d not in black_list:
                        if white_list is None or white_list is not None and d in white_list:
                            subjects.append(dname)
                        break

    voxel_spacings = {}
    for subject in subjects:
        dir_path = os.path.join(dicom_dir, subject)
        print('Converting %s' % subject)
        num = len(os.listdir(dir_path))
        if os.path.isfile(dir_path) or num <= 0:
            continue
        img, voxel_spacing = read_subject(subject)
        outfile = os.path.join(npz_dir, '%s.npz' % subject)

        # delete if exits
        if os.path.exists(outfile):
            os.remove(outfile)

        print('Image Size: {}'.format(img.shape))
        np.savez(outfile, vol_data=img, voxel_spacings=voxel_spacing)
        voxel_spacings[subject] = voxel_spacing

    with open(os.path.join(npz_dir, 'voxel_spacings.json'), 'w') as fp:
        json.dump(voxel_spacings, fp)


def recommend_method(fd: str):
    if fd.endswith(tuple(SUFFIX)):
        return read_image
    else:
        return read_one_voxel


def convert2numpy(data):
    if ants_exits and isinstance(data, ants.ANTsImage):
        meta_data = {
            'spacing': data.spacing,
            'origin': data.origin,
            'direction': data.direction,
            'orientation': data.orientation
        }
        return data.numpy(), meta_data
    elif isinstance(data, dataset.FileDataset):
        meta_data = data.file_meta.to_json_dict()
        return np.array(data.pixel_array), meta_data
    elif isinstance(data, dicomdir.DicomDir):
        meta_data = data.file_meta.to_json_dict()
        return np.array(data.pixel_array), meta_data
    elif isinstance(data, np.ndarray):
        return data, None
    elif isinstance(data, figure.Figure):
        return plot.fig2data(data), None
    elif sitk_exits and isinstance(data, sitk.Image):
        return sitk.GetArrayFromImage(data), None
    elif isinstance(data, PIL.Image.Image):
        # data = data.convert()
        # todo test
        return np.asarray(data.getdata(), dtype='float64'), None
    else:
        raise RuntimeError('Unknown data format!')


def compute_orientation(init_axcodes, final_axcodes):
    """
    A thin wrapper around ``nib.orientations.ornt_transform``

    :param init_axcodes: Initial orientation codes
    :param final_axcodes: Target orientation codes
    :return: orientations array, start_ornt, end_ornt
    """
    ornt_init = nib.orientations.axcodes2ornt(init_axcodes)
    ornt_fin = nib.orientations.axcodes2ornt(final_axcodes)

    ornt_transf = nib.orientations.ornt_transform(ornt_init, ornt_fin)

    return ornt_transf, ornt_init, ornt_fin


# reorientation to R A I
# 默认轴标签为左（L）、右（R）、后（P）、前（A）、下（I）、上（S）。
# 创建以下变换以将体积重新定向为“右、前、下”（RAI）方向。
def reorientation_RAI(data_array, meta_data: dict):
    """
    :param data_array: the data to reorientation
    :param meta_data: the information of the data
    :return: the reorientated data array
    """
    to_ = 'RAI'
    ort = meta_data.get('orientation')
    assert ort is not None
    meta_data['orientation'] = to_

    return do_reorientation(data_array, ort, to_)


def reorientation_PLI(data_array, meta_data: dict):
    to_ = 'PLI'
    ort = meta_data.get('orientation')
    assert ort is not None
    meta_data['orientation'] = to_

    return do_reorientation(data_array, ort, to_)


def do_reorientation(data_array, init_axcodes, final_axcodes):
    """
    source: https://niftynet.readthedocs.io/en/dev/_modules/niftynet/io/misc_io.html#do_reorientation
    Performs the reorientation (changing order of axes)
    Help in Chinese:
        # 轴标签对应为左（L）、右（R）、后（P）、前（A）、下（I）、上（S）。
        # Example: “右、前、下”（RAI）

    :param data_array: 3D Array to reorient
    :param init_axcodes: Initial orientation
    :param final_axcodes: Target orientation
    :return data_reoriented: New data array in its reoriented form
    """
    ornt_transf, ornt_init, ornt_fin = compute_orientation(init_axcodes, final_axcodes)
    if np.array_equal(ornt_init, ornt_fin):
        return data_array

    return nib.orientations.apply_orientation(data_array, ornt_transf)


# for MRI images
def n4_bias_field_correction_ants(mr_image, convert_numpy=True):
    def convert2ants_image(img):
        if isinstance(img, dataset.FileDataset) or isinstance(img, dicomdir.DicomDir) or isinstance(img,
                                                                                                    ants.ANTsImage):
            return img
        else:
            np_image, _ = convert2numpy(img)
            np_image = np_image.astype('float32')
            ants_image = ants.from_numpy(np_image)

        return ants_image

    mr_image = convert2ants_image(mr_image)
    mr_image = ants.n4_bias_field_correction(mr_image)
    return mr_image if not convert_numpy else convert2numpy(mr_image)


def n4_bias_field_correction_sitk(mr_data, convert_numpy=True):
    def convert2sitk_image(img):
        if isinstance(img, sitk.Image):
            return img
        else:
            np_image, _ = convert2numpy(img)
            np_image = np_image.astype('float32')
            sitk_image = sitk.GetImageFromArray(np_image)
        return sitk_image

    mr_data = convert2sitk_image(mr_data)
    maskImage = sitk.OtsuThreshold(mr_data, 0, 1, 200)
    inputImage = sitk.Cast(mr_data, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([10] * 4)
    print(corrector)
    output_mr = corrector.Execute(inputImage, maskImage)

    return output_mr if not convert_numpy else convert2numpy(output_mr)[0]


# code resourse: https://www.atyun.com/23342.html
def resample_sitk(sitk_image: sitk.Image, out_spacing=None, is_label=False):
    # Resample images to 2mm spacing with SimpleITK
    if out_spacing is None:
        out_spacing = [1.0, 1.0, 1.0]
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(sitk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(sitk_image)


def resample(scans_stack, scan=None, old_spacing: tuple = None, new_spacing=(1, 1, 1), interpolation=0, order=3,
             mode='constant'):
    return du.resample(scans_stack, scan=scan, old_spacing=old_spacing, new_spacing=new_spacing,
                       interpolation=interpolation, order=order, mode=mode)


def resample_nd(np_voxel, spacing: tuple = None, new_spacing=(1, 1, 1), interpolation='linear'):
    """
    :param np_voxel: numpy N-d voxel array.
    :param spacing: the spacing of the voxel.
    :param new_spacing: the except spacing.
    :param interpolation: the interpolation.
    :param new_spacing
    :return: the resampled voxel with
    """
    spacing = np.array(spacing)
    new_spacing = np.array(new_spacing)

    resize_factor = spacing / new_spacing
    new_real_shape = np_voxel.shape * resize_factor
    new_shape = np.round(new_real_shape).astype('int')
    real_resize_factor = new_shape / np_voxel.shape
    new_spacing = spacing / real_resize_factor

    transformed_loc = _get_transformed_locs(np_voxel.shape, new_shape)
    output = _interpn(transformed_loc, np_voxel.shape, np_voxel, new_shape, interpolation)

    # the function is like below codes, too slow on CPU
    # for locs, tlocs in zip(local_list, transformed_loc):
    #     output[int(locs[0]), int(locs[1]), int(locs[2])] = np_voxel[int(tlocs[0]), int(tlocs[1]), int(tlocs[2])]

    return output, new_spacing


try:
    import tps
except:
    print("No TPS lib installed :)")


def resample_nd_by_tps(np_voxel: np.ndarray, ctrl_points, trans_ctrl_points, interpolation='nearest'):
    """
    A simple TPS algorithm implement from C++, see {@link tps.py}, this function also use the N-D resample by vx, please see
    {@link _interpn}
    """
    trans = tps.TPS(ctrl_points, trans_ctrl_points)
    grid = np.array(_grid(np_voxel))
    shape = grid.shape
    location_list = grid.reshape(shape[0], -1).transpose()
    transformed_loc = trans(location_list)
    return _interpn(transformed_loc, np_voxel.shape, np_voxel, np_voxel.shape, interpolation)


def resample_nd_by_transform_field(np_voxel: np.ndarray, transformed_loc, interpolation):
    """
    Resample N-D data by linear or nearest resample method qucikly use the numpy array without python 'for'
    But the linear one still has bug when upsample NODE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    :param np_voxel: the data to resample
    :param transformed_loc: the pxiel-level displacement to warp the data.
    :param interpolation: the method of interpolation to use, now support nearest and linear.
    :return: resampled voxel/image/N-D data
    """
    # assert np_voxel.shape == transformed_loc.shape[:-1]
    return _interpn(transformed_loc, np_voxel.shape, np_voxel, np_voxel.shape, interpolation)


def _interpn(loc, org_size, vox, new_shape, interpolation):
    def prod_n(lst):
        prod = lst[0]
        for p in lst[1:]:
            prod *= p
        return prod

    def sub2ind(siz, subs, **kwargs):
        """
        assumes column-order major
        """
        # subs is a list
        assert len(siz) == len(subs), 'found inconsistent siz and subs: %d %d' % (len(siz), len(subs))

        k = np.cumprod(siz[::-1])

        ndx = subs[-1]
        for i, v in enumerate(subs[:-1][::-1]):
            ndx = ndx + v * k[i]

        return ndx

    interp_vol = 0
    loc = loc.astype(np.float64)
    # second order or first order
    if interpolation == 'linear':
        nb_dims = len(org_size)
        loc0 = np.floor(loc)

        # clip values
        max_loc = [d - 1 for d in list(org_size)]
        clipped_loc = [np.clip(loc[..., d], 0, max_loc[d]) for d in range(nb_dims)]
        loc0lst = [np.clip(loc0[..., d], 0, max_loc[d]) for d in range(nb_dims)]

        # get other end of point cube
        loc1 = [np.clip(loc0lst[d] + 1, 0, max_loc[d]) for d in range(nb_dims)]
        locs = [[f.astype(np.int32) for f in loc0lst], [f.astype(np.int32) for f in loc1]]

        #
        # compute the difference between the upper value and the original value
        # differences are basically 1 - (pt - floor(pt))
        #   because: floor(pt) + 1 - pt = 1 + (floor(pt) - pt) = 1 - (pt - floor(pt))
        diff_loc1 = [loc1[d] - clipped_loc[d] for d in range(nb_dims)]
        diff_loc0 = [1 - d for d in diff_loc1]

        weights_loc = [diff_loc1, diff_loc0]  # note reverse ordering since weights are inverse of diff.

        # go through all the cube corners, indexed by a ND binary vector
        # e.g. [0, 0] means this "first" corner in a 2-D "cube"
        cube_pts = list(itertools.product([0, 1], repeat=nb_dims))
        interp_vol = 0
        vox_reshaped = np.reshape(vox, (-1, org_size[-1]))
        for c in cube_pts:
            # get nd values
            # note re: indices above volumes via https://github.com/tensorflow/tensorflow/issues/15091
            #   It works on GPU because we do not perform index validation checking on GPU -- it's too
            #   expensive. Instead we fill the output with zero for the corresponding value. The CPU
            #   version caught the bad index and returned the appropriate error.
            subs = [locs[c[d]][d] for d in range(nb_dims)]

            # tf stacking is slow for large volumes, so we will use sub2ind and use single indexing.
            # indices = tf.stack(subs, axis=-1)
            # vol_val = tf.gather_nd(vol, indices)
            # faster way to gather than gather_nd, because the latter needs tf.stack which is slow :(
            idx = sub2ind(org_size, subs)
            vol_val = np.take(vox_reshaped, idx)

            # get the weight of this cube_pt based on the distance
            # if c[d] is 0 --> want weight = 1 - (pt - floor[pt]) = diff_loc1
            # if c[d] is 1 --> want weight = pt - floor[pt] = diff_loc0
            wts_lst = [weights_loc[c[d]][d] for d in range(nb_dims)]
            # tf stacking is slow, we we will use prod_n()
            # wlm = tf.stack(wts_lst, axis=0)
            # wt = tf.reduce_prod(wlm, axis=0)
            wt = prod_n(wts_lst)
            # wt = np.expand_dims(wt, -1)

            # compute final weighted value for each cube corner
            interp_vol += wt * vol_val

        interp_vol = np.reshape(interp_vol, new_shape)

    elif interpolation == 'nearest':
        loc = np.round(loc).astype(np.int32)

        shape = loc.shape
        a = np.empty_like(loc)
        for i in range(shape[-1]):
            np.clip(loc[..., i], 0, org_size[i] - 1, a[..., i])

        a = tuple(a.transpose())
        interp_vol = vox[a].reshape(new_shape)

    return interp_vol


def _get_transformed_locs(shape, new_shape):
    grid = grid2(new_shape)
    ndims = len(shape)
    real_resize_factor = shape / new_shape

    scales = np.array(list(real_resize_factor))

    # we only do scale no rotation or transport
    scales = np.multiply(scales, np.eye(ndims))

    # get the original coordinate DIMS x PIXEL_NUMBER
    grid = np.array(grid)
    shape = grid.shape
    location_list = grid.reshape(shape[0], -1).transpose()

    # the location in the origin voxel
    transformed_loc = _transform(location_list, scales).transpose()

    return transformed_loc.transpose()


def _transform(loc, affine_matrix):
    return np.matmul(loc, affine_matrix)


def _grid(np_voxel: np.ndarray):
    sp = np_voxel.shape
    return _grid2(sp)


def grid2(shape):
    ndims = len(shape)
    shape = np.array(shape, dtype='uint')
    axis_list = [[i for i in range(shape[dims])] for dims in range(ndims)]
    grid = np.meshgrid(*axis_list, indexing='ij')
    return grid


def resample_ants(image, resample_params, use_voxels=False, interp_type=1):
    return ants.resample_image(image, resample_params, use_voxels=use_voxels, interp_type=interp_type)


def center_trans_sitk(voxel_moving: np.ndarray, voxel_fixed: np.ndarray, filled_value=0.0, sitk_inter=None):
    if not sitk_inter: sitk_inter = sitk.sitkLinear
    voxel_fixed = sitk.GetImageFromArray(voxel_fixed)
    voxel_moving = sitk.GetImageFromArray(voxel_moving)

    inital_trans = sitk.CenteredTransformInitializer(voxel_fixed,
                                                     voxel_moving,
                                                     sitk.Euler3DTransform(),
                                                     sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)

    registration_method.SetOptimizerAsExhaustive(numberOfSteps=[0, 1, 1, 0, 0, 0], stepLength=np.pi)
    registration_method.SetOptimizerScales([1, 1, 1, 1, 1, 1])

    registration_method.SetInitialTransform(inital_trans, inPlace=True)
    registration_method.Execute(voxel_fixed, voxel_moving)

    moved_voxel = sitk.Resample(voxel_moving, voxel_fixed, inital_trans, sitk_inter, filled_value, voxel_moving.GetPixelID())

    return convert2numpy(moved_voxel)[0]


# todo for instance normalize, MRI in different organs has different pixel
def min_max_normalized(scans, mode='MRI', min_bound=-1000, max_bound=3000):
    if mode == 'CT':
        scans = (scans - min_bound) / (max_bound - min_bound)
    elif mode == 'MRI':
        scans = (scans - min_bound) / (max_bound - min_bound)

    return scans


def bound_normalized(scans: np.ndarray):
    min_bound = scans.min()
    max_bound = scans.max()
    scans = (scans - min_bound) / (max_bound - min_bound + np.e)

    return scans


# for single channel images
# for multi channel ones, we need to spilt that at the channel dimension
def statics_mean_std(file_list: List[str], **kwargs):
    vol_var = kwargs.pop('vol_var')

    n = len(file_list)
    assert n > 0
    mean_sum = 0
    max_val = -1e6
    min_val = 1e6
    for path in file_list:
        method = recommend_method(path)

        if vol_var is not None:
            np_data = read_one_voxel(path, var_data=vol_var)
        else:
            np_data = method(path)

        mean_sum += np.mean(np_data)
        max_val = max(np.max(np_data), max_val)
        min_val = min(np.min(np_data), min_val)

    mean = mean_sum / n

    var_sum = 0
    for path in file_list:
        method = recommend_method(path)

        if vol_var is not None:
            np_data = read_one_voxel(path, var_data=vol_var)
        else:
            np_data = method(path)

        error = (np_data - mean)
        var_sum += np.mean(error * error)

    std = math.sqrt(var_sum / n)  # has bais

    return mean, std, max_val, min_val


# the different of this and the above function is that it statistic pixel-level.
def statistics_mean_std2(file_list: List[str], **kwargs):
    vol_var = kwargs.pop('vol_var')

    n = len(file_list)
    assert n > 0
    mean = 0
    max_val = -1e6
    min_val = 1e6
    for path in file_list:
        method = recommend_method(path)

        if vol_var is not None:
            np_data = read_one_voxel(path, var_data=vol_var)
        else:
            np_data = method(path)

        mean += np_data / n
        max_val = np.maximum(max_val, np_data)
        min_val = np.minimum(max_val, np_data)

    var = 0
    for path in file_list:
        method = recommend_method(path)

        if vol_var is not None:
            np_data = read_one_voxel(path, var_data=vol_var)
        else:
            np_data = method(path)

        error = (np_data - mean)
        var += error * error / n

    return mean, np.sqrt(var), max_val, min_val


def voxels_statistics(np_data: np.ndarray, do_print=True, statics_voxel_class=False):
    mu = np.mean(np_data)
    sum_ = np.sum(np_data)
    std = np.std(np_data)
    max_ = np.max(np_data)
    min_ = np.min(np_data)

    classes = None
    class_num = -1

    if do_print:
        print('mean:', mu, ' sum:', sum_, ' std:', std, ' max:', max_, ' min:', min_)

    if statics_voxel_class:
        flt = np_data.flatten()
        classes = set(flt)
        class_num = len(classes)
        if do_print:
            print('classes:', classes, 'number:', class_num)

    return mu, sum_, std, max_, min_, classes, class_num


def get_overall_bounding_box(b_list: List[Dict[float, List[Tuple]]], label):
    if len(b_list) <= 0:
        return None

    ndims = len(list(b_list[0].values())[0])

    min_list = [1e6] * ndims
    max_list = [-1] * ndims

    for v_b in b_list:
        bb = v_b[label]
        for d in range(ndims):
            x_d = bb[d]
            min_list[d] = min(x_d[0], min_list[d])
            max_list[d] = max(x_d[1], max_list[d])

    return tuple(min_list), tuple(max_list)


def bounding_box_shapes_statistics(b_list: List[Dict[float, List[Tuple]]], label):
    if len(b_list) <= 0:
        return None

    ndims = len(list(b_list[0].values())[0])

    shapes = []
    for v_b in b_list:
        bb = v_b[label]
        shape = []
        start = []
        for d in range(ndims):
            x_d = bb[d]
            d_s = x_d[1] - x_d[0]
            shape.append(d_s)
            start.append(x_d[0])
        shapes.append((tuple(start), tuple(shape)))

    return shapes


def gt_bounding_box_statistics(gt_path, labels: Tuple[float], file_pattern='*', var_data='vol'):
    """
    To statistics all Ground Truth Voxels that the minimal bounding box to around them. the path should be a DIR of some
    voxels formatted of 'nii', 'nii.gz', 'npy', 'npz', 'mgz'
    :param file_pattern: files to statics in the
    :param gt_path
    :param labels: the labels to statics
    :param var_data: if read a NPZ file, the array's key to extract the array
    :param gt_path: the directory of a collection of some voxels
    :return: a label-dict of bounding box (like [(x_1_min, x_1_max),(x_2_min, x_2_max), ..., (x_n_min, x_n_max)]) with
    different labels
    """
    result = []
    files = glob.glob(os.path.join(gt_path, file_pattern))
    if len(labels) <= 0:
        raise RuntimeError('Labels need at least one dim')

    for f in files:
        np_voxel = read_one_voxel(f, var_data=var_data)
        r = {}
        for label in labels:
            res = _gt_bounding_box(np_voxel, label)
            r[label] = res
        result.append(r)

    return result


def _gt_bounding_box(gt_voxel: np.ndarray, label):
    temp = np.where(gt_voxel == label, 1, 0)
    ndims = temp.ndim
    shape = temp.shape

    # to statics at every dim
    bounding_box = []
    rr = range(ndims)
    dims = set(rr)
    for dim in dims:
        dim_set = set()
        dim_set.add(dim)
        f_vx: np.ndarray = temp.sum(axis=tuple(dims - dim_set))

        min_, max_ = -1, -1
        r = range(shape[dim])
        for i in r:
            if f_vx[i] > 0:
                min_ = i
                break
        if min_ != -1:
            for i in reversed(r):
                if f_vx[i] > 0:
                    max_ = i
                    break

        s = (min_, max_)
        bounding_box.append(s)

    return bounding_box


def normalized(image, mean, std):
    return (image - mean) / std


def combine_labels(gt, labels=(), in_=1, out_=0):
    a = gt != gt
    for i in labels:
        a |= (gt == i)
    return np.where(a, in_, out_)


def crop_by_gt(np_voxel, crop_size: Tuple, np_gt=None, label=255, pad_mode='constant', **kwargs):
    def get_center(np_gt, label):
        bb = _gt_bounding_box(np_gt, label)
        ndims = len(bb)
        center = []
        for d in range(ndims):
            x_d = bb[d]
            d_s = x_d[0] + (x_d[1] - x_d[0]) // 2
            center.append(d_s)
        return center

    if np_gt is not None:
        center = np.array(get_center(np_gt.copy(), label))
    else:
        center = np.array(np_voxel.shape) // 2

    # crop by center
    np_cs = np.array(crop_size)
    half = np_cs // 2
    another_half = np_cs - half
    start = center - half
    end = center + another_half

    ndims = start.shape[0]
    slices = ()
    shape = np_voxel.shape
    # check crop
    # if do_padding, padding the edges of every dim
    for d in range(ndims):
        start_d = start[d]
        end_d = end[d]
        padding_left = 0
        padding_right = 0
        if start_d < 0:
            padding_left = 0 - start_d
        if end_d > shape[d]:
            padding_right = end_d - shape[d]

        start_d += padding_left
        end_d += padding_left

        # create padding tuple at dim of d
        padding_tuple = numpy_util.create_i_dim_padding((padding_left, padding_right), d, ndims)
        if padding_left + padding_right > 0:
            print('Warning: do padding ({}, {}) at {}\'th axis of the voxel to crop completely.'.format(padding_left,
                                                                                                        padding_right,
                                                                                                        d))
            np_voxel = np.pad(np_voxel, padding_tuple, mode=pad_mode, **kwargs)

        slices += (slice(start_d, end_d),)

    return np_voxel[slices]


def padding():
    pass


def center_pad(data, to_size, mode='constant', constant_values=(0, 0)):
    delta_shape = [max(j - i, 0) for i, j in zip(list(data.shape), to_size)]
    f_pad = [d // 2 for d in delta_shape]
    b_pad = [d - d // 2 for d in delta_shape]
    dim_pad = [(i, j) for i, j in zip(f_pad, b_pad)]
    return np.pad(data, dim_pad, mode=mode, constant_values=constant_values)


def center_crop(data, to_size):
    delta_shape = [max(j - i, 0) for i, j in zip(to_size, list(data.shape))]
    f_pad = [d // 2 for d in delta_shape]
    b_pad = [d - d // 2 for d in delta_shape]

    dim_slice = tuple([slice(i, -j) if j != 0 else slice(None) for i, j in zip(f_pad, b_pad)])
    return data[dim_slice]


def voxel_minimum_size_statistics(files, vol='vol'):
    minimum_shape = None

    for f in files:
        voxel = recommend_method(f)(f)
        if not minimum_shape:
            minimum_shape = list(voxel.shape)
        else:
            minimum_shape = [min(minimum_shape[d], voxel.shape[d]) for d in range(len(minimum_shape))]

    return minimum_shape


def voxel_ct_normalized(files, labels, sample_count=20):
    files.sort()
    labels.sort()
    cnt = sample_count
    voxel_statistics = dict()
    min_, max_ = 1e7, -1e7
    for file, label in zip(files, labels):
        if sample_count == 0:
            break
        sample_count -= 1
        voxel = recommend_method(file)(file)
        label_voxel = recommend_method(label)(label)
        masked_voxel = (voxel * label_voxel)

        min_ = min(int((masked_voxel + (1 - label_voxel) * 1e10).min()), min_)
        max_ = max(int((masked_voxel - (1 - label_voxel) * 1e10).max()), max_)

    for i in range(min_, max_ + 1, 1):
        voxel_statistics[i] = 0

    n = 0
    sample_count = cnt
    for file, label in zip(files, labels):
        if sample_count == 0:
            break
        sample_count -= 1

        voxel = recommend_method(file)(file)
        label_voxel = recommend_method(label)(label)
        masked_voxel = voxel * label_voxel + (1 - label_voxel) * (max_ + 7)
        masked_voxel = masked_voxel.astype(np.int64)

        for i in range(min_, max_ + 1, 1):
            c = np.sum(masked_voxel == i)
            voxel_statistics[i] += c >> n
            if voxel_statistics[i] * (max_ - min_) > (1 << 64 - 1):
                n += 1
                for j in range(min_, max_ + 1, 1):
                    voxel_statistics[j] = voxel_statistics[j] >> 1

    total_count = 0

    for k, v in voxel_statistics.items():
        total_count += v

    f, b = 0.05, 0.995
    cur = 0
    f_v, b_v = -1, -1
    mu = 0
    for i in range(min_, max_ + 1):
        voxel_statistics[i] = voxel_statistics[i] / total_count
        cur += voxel_statistics[i]
        mu += voxel_statistics[i] * i
        if cur > f and f_v == -1:
            f_v = i
        elif cur > b and b_v == -1:
            b_v = i

    var = 0
    for i in range(min_, max_ + 1):
        var += (i - mu) ** 2 * voxel_statistics[i]

    return mu, var, f_v, b_v


def take_slice(vol: np.ndarray, mode: int = 0, rot=(0, 1, -1), take_fun=None):
    """
    :param mode: the mode to take one slice at each axis; number 0 will take the middle of the voxels, 1 means the first
    and 2 will take the last one slice.
    :param take_fun: the pick function of the slice of the volume that it will pick.
    :param rot: the rotation of different axises
    :param vol: the 3d-array(may be a voxel)
    :return: the list of the 3d-vol by 3 axis
    """

    def take_middle(shape_len):
        return shape_len // 2

    def take_first(shape_len):
        return 0

    def take_last(shape_len):
        return shape_len - 1

    ndims = len(vol.shape)
    assert ndims == 3 and len(rot) == 3

    if take_fun is None:
        take_fun = take_middle

    if mode == 0:
        take_fun = take_middle
    elif mode == 1:
        take_fun = take_first
    elif mode == 2:
        take_fun = take_last
    else:
        pass

    mid_slices = [np.take(vol, take_fun(vol.shape[d]), axis=d) for d in range(ndims)]

    for i in range(0, ndims):
        mid_slices[i] = np.rot90(mid_slices[i], rot[i])

    return mid_slices


def fusion(datas: List[np.ndarray], mode='cover', src_a=.5, dst_a=.5):
    # mode: cover, max, min, multiply, add
    m = datas[0].copy()

    for i in range(1, len(datas)):
        if mode == 'cover':
            mask = datas[i] != 0
            m[mask] = 0
            m += datas[i]
        elif mode == 'max':
            m = np.maximum(m, datas[i])
        elif mode == 'min':
            m = np.minimum(m, datas[i])
        elif mode == 'multiply':
            m = np.multiply(m, datas[i])
        elif mode == 'add':
            m = np.add(m * src_a, datas[i] * dst_a)

        else:
            raise RuntimeError('Unknown Mode: {}'.format(mode))

    return m


def png2npz2(root, file_path, file_name: str, is_dir: bool, **kwargs):
    """
    module of aggregation's process function, see {@link aggregation.py},
    :param root: start root folder
    :param file_path: file_path
    :param file_name: file_name
    :param is_dir: is_dir
    :param kwargs: kwargs
    :return: npz file
    """
    target_dir = os.path.abspath(kwargs.pop("target_dir"))
    root = os.path.abspath(root)
    file_path = os.path.abspath(file_path)
    pattern = kwargs.pop("pattern")
    target = kwargs.pop("target")

    rel_path = os.path.relpath(file_path, root)
    target = file_name if target == "name" else file_path
    if not is_dir:
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        if re.match(pattern, target) and os.path.exists(target_dir):
            to = os.path.join(target_dir, rel_path.replace('png', 'npz'))
            image = read_image(file_path)
            write2npz(image, to)


def range_patch(start: Tuple, shape: Tuple, sample_size: Tuple, sample_stride: Tuple):
    end = []
    for i, j, k in zip(shape, sample_size, sample_stride):
        al = (i - j) % k == 0
        end.append((i - j) // k + (1 if not al else 0))
    return _range_patch(start, tuple(end), sample_size)


def _range_patch(start: Tuple, end: Tuple, sample_stride: Tuple):
    pass
    # return (a, b, c for a, b, c in zip(range(start[0], end[0], sample_stride[0]), range(start[1], end[1], sample_stride[1]), range(start[2], end[2], sample_stride[2])))


def joint_patch_by_pos(start: Tuple, shape: Tuple, patch: np.ndarray, skip_dims=2, count_data=None, output_data=None):
    sample_size = patch.shape
    skip_slices = skip_dims * (slice(None),)
    end = [i + j for i, j in zip(start, sample_size[skip_dims:])]
    slices = tuple([slice(s, e) for s, e in zip(start, end)])
    if count_data is None:
        count_data = np.zeros(output_data.shape)
    if output_data is None:
        output_data = np.zeros(shape)
    slices = skip_slices + slices
    count_data[slices] += 1
    output_data[slices] += patch[skip_slices]

    return output_data, count_data


def make_patches_3d(voxel: np.ndarray, size_: Tuple, stride, save_dir, post_fix='.patch', random_sample=False):
    shape_ = voxel.shape
    sample_size = size_

    assert len(sample_size) == 3 and len(shape_) in [3, 4]

    _, (x_step, y_step, z_step) = get_patch_number(voxel, shape_, stride)

    if not random_sample:
        for x in range(x_step):
            for y in range(y_step):
                for z in range(z_step):
                    start_x = x * stride[0]
                    start_y = y * stride[1]
                    start_z = z * stride[2]
                    if len(shape_) == 4:
                        patch = voxel[:, start_x: start_x + sample_size[0], start_y: start_y + sample_size[1],
                                start_z: start_z + sample_size[2]]
                    else:
                        patch = voxel[start_x: start_x + sample_size[0], start_y: start_y + sample_size[1],
                                start_z: start_z + sample_size[2]]

                    if patch:
                        write2nii(patch, os.path.join(save_dir, '_{}_{}_{}{}'.format(x, y, z, post_fix)))
    else:
        pass


def get_patch_number(voxel_shape, patch_shape: Tuple, patch_stride: Tuple):
    shape_ = voxel_shape

    shape_ = [(shape_[i] - patch_shape[i]) // patch_stride[i] + 1 for i in range(len(shape_))]
    total_size = 1
    for i in shape_:
        total_size *= i
    return total_size, shape_


def sample_patch_by_index(voxel: np.ndarray, index, sample_size: Tuple, sample_stride: Tuple,
                          sample_number_dims: Tuple):
    res = get_position_by_index(index, sample_number_dims)
    assert res and len(res) == len(sample_stride)
    res = [i * n for i, n in zip(res, sample_stride)]
    return sample_patch_by_position(voxel, tuple(res), sample_size)


def sample_patch_by_position(voxel: np.ndarray, sample_start_position: Tuple, sample_size: Tuple):
    end = [i + j for i, j in zip(sample_start_position, sample_size)]
    slices = tuple([slice(s, e) for s, e in zip(sample_start_position, end)])

    start_x, start_y, start_z = sample_start_position
    shape_ = voxel.shape
    skip_slices = len(voxel.shape) - len(sample_size)
    assert skip_slices >= 0

    patch = voxel[skip_slices * (slice(None),), slices]

    return patch


def get_position_by_index(index: int, sample_number_dims: Tuple) -> Tuple:
    # sample plan:
    """
    For a dataset which sample count is N, each data will be cropped as M patches, each one is sampled from the
    origin voxel at the index of (i, j, k). The upper boundings of i, j, k are I, J, K that M = I * J * K
    Now we have a patches dataset that size is N * M, just traverse it, and get a index 'x' (0 <= x < N * M)
    for x, we get the sample index y = x % N, and to get the actual coordinate we need the upper bounding I, J, K.
    """
    s = len(sample_number_dims)
    if s == 1:
        return (index,)

    divs = []
    cur: int = 1
    for i in sample_number_dims[:-1]:
        cur *= i
        divs.append(cur)

    res_indexs = []
    cur = index
    for i in reversed(divs):
        idx = cur // i
        cur = cur % i
        res_indexs.append(idx)
    res_indexs.append(cur)

    return tuple(reversed(res_indexs))


def joint_patches(output_data: np.ndarray, patch_stride: Tuple, patches: List[Tuple], skip_dims=2):
    """
    Details to see {@link joint_patch}.
    """

    count = None
    for index, patch in patches:
        output_data, count = joint_patch(output_data, patch, patch_stride, index, skip_dims, count)

    count[count == 0] = 1

    return output_data / count, count


def joint_patch(output_data: np.ndarray, patch: np.ndarray, patch_stride: Tuple, index: int, skip_dims=2,
                count_data: np.ndarray = None):
    """
    The function is to joint patches in order. To joint the patches, we first to create a empty dst numpy array, then fill the pixel into
    the white room, and use a count matrix (here, we call it count data) to count the pixel[for example, at (x, y, z, ...)] be filled how
     many times. At last we calculate the avg pixel-level value for all pixels: output data / count data.

    :param output_data: the result of the jpint data.
    :param patch: one patch to fill in.
    :param patch_stride: the patch stride
    :param index: the patch's index
    :param skip_dims: wheath the dimension to ski, for example the channel or the batchsize dimension
    :param count_data: the support data struct to save the counts
    :return: result of jointed voxel/image
    """

    shape_ = output_data.shape
    patch_shape = patch.shape
    assert len(shape_) == len(patch_shape)

    skip_slices = skip_dims * (slice(None),)

    _, boundings = get_patch_number(shape_[skip_dims:], patch_shape[skip_dims:], patch_stride)
    start = get_position_by_index(index, tuple(boundings))
    start = [i * n for i, n in zip(patch_stride, start)]
    end = [i + j for i, j in zip(start, patch_shape[skip_dims:])]
    slices = tuple([slice(s, e) for s, e in zip(start, end)])
    if count_data is None:
        count_data = np.zeros(output_data.shape)
    slices = skip_slices + slices
    count_data[slices] += 1
    output_data[slices] += patch[skip_slices]
    return output_data, count_data
