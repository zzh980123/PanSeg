import json
import os
from typing import List, Dict, Any
import numpy as np

from cv2 import cv2


class BBox:

    def __init__(self, _id: int, map_file: str, bbox, _type: int):
        self.id = _id
        self.map_file = map_file
        self.bbox = bbox
        self.extra = None
        self.type = _type


class ROIParser:

    def parse(self, text) -> List[BBox]:
        return list()

    def get_type(self) -> int:
        return 1


class JsonParser(ROIParser):

    def __init__(self, file_flag: str, box_flag: str, ann_flag=None):
        self.ann_flag = ann_flag
        self.file_flag = file_flag
        self.box_flag = box_flag

    def parse(self, text) -> List[BBox]:
        text = json.loads(text)
        roi_list = list()
        iid: int = 0

        if self.ann_flag:
            ann_list: [str, Any] = text[self.ann_flag]

            for ann in ann_list:
                box = self.create_roi(ann, iid)
                if box:
                    roi_list.append(box)
                    iid += 1
            return roi_list
        else:
            for ann in text:
                box = self.create_roi(ann, iid)
                if box:
                    roi_list.append(box)
                    iid += 1
            return roi_list

    def create_roi(self, ann, iid: int) -> BBox:
        return BBox(iid, ann[self.file_flag], self.box_flag, self.get_type())


class Cutter:

    def __init__(self, parser: ROIParser):
        self.bboxs: Dict[str, BBox] = dict()
        self.parser = parser

    def __get_roi(self, text) -> List[BBox]:
        return self.parser.parse(text) if self.parser else list()

    def __collect_roi_list(self, rio_info_file: str) -> List[BBox]:
        with open(rio_info_file, encoding='utf-8') as file:
            text = file.read()
        return self.__get_roi(text)

    def __do_cut(self, images, target_dir: str) -> None:
        for image in images:
            img = cv2.imread(images, 'rw')
            box: BBox = self.bboxs.get(image)
            cropped_images = self.__default_cut(box, img)
            for cropped_image in cropped_images:
                new_name = self._rename(image, target_dir, box)
                cv2.imwrite(new_name, cropped_image)
        return

    def __default_cut(self, box: BBox, img, expand_list=None) -> Any:
        images = list()
        if not expand_list:
            expand_list = {0}
        if box.type == 0:
            x, y, w, h = box.bbox
            for expand in expand_list:
                cropped = img[max(y - expand, 0): min(y + h + expand, img.shape[0]),
                          max(x - expand, 0): min(x + w + expand, img.shape[1])]
                images.append(cropped)
        elif box.type == 1:
            x, y, r = box.bbox
            for expand in expand_list:
                cropped = img[max(y - expand - r, 0): min(y + r + expand, img.shape[0]),
                          max(x - expand - r, 0): min(x + r + expand, img.shape[1])]
                images.append(cropped)
        else:
            images = self._custom_cut(box, img, expand_list)

        return images

    def _custom_cut(self, box: BBox, img, expand_list=None) -> Any:
        return None

    def _get_ori_images(self, image_dir: str, rois: List[BBox]) -> List[str]:
        ori_images = list()
        for box in rois:
            file_path = image_dir + os.sep + box.map_file
            if os.path.exists(file_path):
                ori_images.append(file_path)
                self.bboxs[file_path] = box

        return ori_images

    def _rename(self, old_name: str, target_dir: str, box: BBox) -> str:

        (file_path, whole_file_name) = os.path.split(old_name)
        (file_name, extension) = os.path.splitext(whole_file_name)

        new_name = "{name}{{{label}}}{postfix}".format(name=file_name,
                                                       label=str(box.extra) if box.extra else "",
                                                       postfix=extension)
        return target_dir + os.sep + new_name

    def cut(self, image_dir: str, roi_dir: str):
        rois = self.__collect_roi_list(roi_dir)
        # create dict of file_path and roi
        pro_images = self._get_ori_images(image_dir, rois)
        self.__do_cut(pro_images, roi_dir)


def read_image(image_path: str, var='vol_data'):
    if image_path.endswith('.npy') or image_path.endswith('.npz'):
        image = np.load(image_path)
        if image is None:
            raise FileNotFoundError
        if image_path.endswith('npz'):
            return image[var]
        else:
            return image
    else:
        raise RuntimeError('Only process npy or npz files')


def crop_img_to(img, slices, copy):

    if copy:
        img = img.copy()
    return img[slices]


def get_max_slices(image_set, gt_set):

    def get_foreground(gt_set, background_value=0, tolerance=0.00001):
        first = True
        for image_file in gt_set:
            # 读取image无裁剪，无resize
            image = read_image(image_file)
            # 根据设置的background值,综合所有模态和truth将image data数组中foreground位置设为True
            is_foreground = np.logical_or(image < (background_value - tolerance),
                                          image > (background_value + tolerance))
            if first:
                foreground = np.zeros(is_foreground.shape, dtype=np.uint8)
                first = False

            # 将is_foreground位置像素值设置为1
            foreground[is_foreground] = 1

        return foreground

    def crop_img(img, rtol=1e-8, copy=True, return_slices=False, return_coords=False) -> Any:
        data = img

        infinity_norm = max(-data.min(), data.max())
        passes_threshold = np.logical_or(data < -rtol * infinity_norm,
                                         data > rtol * infinity_norm)

        if data.ndim == 4:
            passes_threshold = np.any(passes_threshold, axis=-1)

        coords = np.array(np.where(passes_threshold))
        start = coords.min(axis=1)
        end = coords.max(axis=1) + 1

        # pad with one voxel to avoid resampling problems
        start = np.maximum(start - 1, 0)
        end = np.minimum(end + 1, data.shape[:3])

        slices = [slice(s, e) for s, e in zip(start, end)]

        if return_coords:
            return start, end

        if return_slices:
            return slices

        return crop_img_to(img, slices, copy=copy)

    coords = list()
    for gt in gt_set:
        gtlist = list()
        gtlist.append(gt)
        fa = get_foreground(gtlist)
        start, end = crop_img(fa, return_coords=True)
        coords.append(start)
        coords.append(end)

        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        #
        # ex = np.where(fa)
        # # 开始绘图
        # fig = plt.figure(dpi=120)
        # ax = fig.add_subplot(111, projection='3d')
        # # 标题
        # plt.title('point cloud')
        # # 利用xyz的值，生成每个点的相应坐标（x,y,z）
        # ax.scatter(ex[0], ex[1], ex[2], c='b', marker='.', s=2, linewidth=0, alpha=1, cmap='spectral')
        # # ax.set_aspect('equal')
        # # ax.axis('scaled')
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # # 显示
        # plt.show()

    coords_np = np.array(coords)
    m_s = coords_np.min(axis=0)
    m_e = coords_np.max(axis=0) + 1
    slices = [slice(s, e) for s, e in zip(m_s, m_e)]

    return slices


def crop_or_pad(img, res_shape=(128, 128, 128)):
    shape = img.shape

    top = int((res_shape[0] - shape[0]) / 2)
    down = res_shape[0] - shape[0] - top

    left = int((res_shape[1] - shape[1]) / 2)
    right = res_shape[1] - shape[1] - left

    front = int((res_shape[2] - shape[2]) / 2)
    back = res_shape[2] - shape[2] - front

    a, b, c, d, e, f = max(0, top), max(0, down), max(0, left), max(0, right), max(0, front), max(0, back)

    padding = ((a, b), (c, d), (e, f))

    pad_img = np.pad(img, padding, 'constant')

    if top < 0:
        a = -top
    else:
        a = 0

    if down < 0:
        b = shape[0] + down
    else:
        b = pad_img.shape[0]

    if left < 0:
        c = -left
    else:
        c = 0

    if right < 0:
        d = shape[1] + right
    else:
        d = pad_img.shape[1]

    if front < 0:
        e = -front
    else:
        e = 0

    if back < 0:
        f = shape[2] + back
    else:
        f = pad_img.shape[2]

    pad_img = pad_img[a:b, c:d, e:f]

    return pad_img
