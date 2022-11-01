import os
from models.model_selector import *

join = os.path.join
import argparse
import numpy as np
import torch
from monai.inferers import sliding_window_inference
import time
import SimpleITK as sitk
import nibabel as nib


def write2nii(array: np.ndarray, file_name, auto_create=True):
    new_image = nib.Nifti1Image(array, np.eye(4))
    nib.save(new_image, file_name)

def maxminNormal(a, outmin, outmax):
    amax = np.max(a)
    amin = np.min(a)
    norm = (a - amin) / (amax - amin)
    output = (norm * (outmax - outmin)) + outmin
    return output

def main():
    parser = argparse.ArgumentParser('predict for pancreatic image segmentation', add_help=False)
    # Dataset parameters
    parser.add_argument('-i', '--input_path', default='./inputs', type=str, help='training data path; subfolders: images, labels')
    parser.add_argument('-l', '--label_path', default='./labels', type=str)
    parser.add_argument("-o", '--output_path', default='./outputs', type=str, help='output path')
    parser.add_argument('--model_path', default='./work_dir/swinunetrv2_3class', help='path where to save models and segmentation results')
    parser.add_argument('--seed', default=2022)
    # parser.add_argument('--show_overlay', required=False, default=False, action="store_true", help='save segmentation overlay')

    # Model parameters
    parser.add_argument('--model_name', default='swinunetrv2', help='select mode: unet, unetr, swinunetr，swinunetrv2')
    parser.add_argument('--num_class', default=1, type=int, help='segmentation classes')
    parser.add_argument('--input_size', default=512, type=int, help='segmentation classes')
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    label_path = args.label_path
    np.random.seed(args.seed)
    os.makedirs(output_path, exist_ok=True)

    img_names = sorted(os.listdir(input_path))
    img_nums = len(img_names)
    img_index = np.arange(img_nums)
    np.random.shuffle(img_index)

    test_rate = 0.2
    test_nums = int(img_nums * test_rate)
    test_index = img_index[:test_nums]

    test_file = [join(input_path, img_names[i]) for i in test_index]
    test_labels = [join(label_path, img_names[i]) for i in test_index]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_factory(args.model_name.lower(), device, args, in_channels=1)

    # find best model
    model_path = join(args.model_path, 'best_Dice_model.pth')

    print(f"Loading {model_path}...")
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    # %%
    roi_size = (args.input_size, args.input_size)
    sw_batch_size = 1
    model.eval()
    # model = model.half()
    torch.set_grad_enabled(False)
    # print(torch.cuda.memory_summary())
    # torch.cuda.set_per_process_memory_fraction(0.5, 0)

    with torch.no_grad():
        for index, test_data in enumerate(test_file, 0):
            # torch.cuda.empty_cache()

            img_data = sitk.GetArrayFromImage(sitk.ReadImage(test_data))  # D H W
            label_data = test_labels[index]
            label_data = sitk.GetArrayFromImage(sitk.ReadImage(label_data))
            d, h, w = img_data.shape
            z = np.any(label_data, axis=(1, 2))
            start_slice, end_slice = np.where(z)[0][[0, -1]]
            pre_nii_data = np.zeros((d, h, w))
            t0 = time.time()
            for i in range(start_slice, end_slice + 1):
                slice_data = img_data[i, :, :]
                slice_data = np.expand_dims(np.expand_dims(slice_data, 0), 0)
                slice_data = maxminNormal(slice_data, 0, 1)
                slice_data = torch.tensor(slice_data, dtype=torch.float).to(device)
                output_img = sliding_window_inference(slice_data, roi_size, sw_batch_size, model)
                output_img = torch.sigmoid(output_img).cpu()
                output_img[output_img > 0.5] = 1
                output_img[output_img < 0.5] = 0
                pre_nii_data[i, :, :] = output_img

            pre_nii_data = np.swapaxes(pre_nii_data, 0, 2)
            img_name = test_data.split('/')[-1]
            write2nii(pre_nii_data, output_path + '/' + img_name)
            t1 = time.time()
            print(f'Prediction finished: {img_name}; img size = {img_data.shape}; costing: {t1 - t0:.2f}s')


if __name__ == "__main__":
    main()
