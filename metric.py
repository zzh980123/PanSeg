from monai.metrics import DiceMetric, HausdorffDistanceMetric
import numpy as np
import argparse
import os
import SimpleITK as sitk
import torch
import csv


def main():
    parser = argparse.ArgumentParser('metric between labels and output', add_help=False)
    parser.add_argument('-i', '--input_path', default='./result/zheyi')
    parser.add_argument('-o', '--output_path', default='./result/zheyi')
    parser.add_argument('-l', '--label_path', default='./dataset/zheyi/labels')
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    label_path = args.label_path

    predict_names = sorted(os.listdir(input_path))
    predict_file = [os.path.join(input_path, predict_names[i]) for i in range(len(predict_names))]
    gt_file = [os.path.join(label_path, predict_names[i]) for i in range(len(predict_names))]

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd_metric = HausdorffDistanceMetric()

    save_file = output_path + '/measure.csv'
    header = ['id', 'DSC', 'HD']

    with open(save_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(predict_file)):
            pred = sitk.GetArrayFromImage(sitk.ReadImage(predict_file[i]))
            gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_file[i]))
            pred = np.swapaxes(np.expand_dims(pred, 0), 1, 3)
            gt = np.swapaxes(np.expand_dims(gt, 0), 1, 3)
            gt.dtype = np.int16
            pred = torch.Tensor(pred)
            gt = torch.Tensor(gt)
            dice = dice_metric(y_pred=pred, y=gt)
            # hd = hd_metric(y_pred=pred, y=gt)
            dice_score = dice_metric.aggregate().item()
            # hd_score = hd_metric.aggregate().item()
            dice_metric.reset()
            id = predict_names[i].split('.')[0]
            # data = [id, dice_score, hd_score]
            data = [id, dice_score]
            writer.writerow(data)
            print(id)

    print('complete!')


if __name__ == '__main__':
    main()
