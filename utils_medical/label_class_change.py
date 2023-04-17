import os
import SimpleITK as sitk

def change_label(raw_label_path, changed_label_path, label_name):
    if not os.path.exists(changed_label_path):
        os.makedirs(changed_label_path)
    cnt = 1
    for nii_file in os.listdir(raw_label_path):
        raw_nii = sitk.ReadImage(os.path.join(raw_label_path, nii_file), sitk.sitkInt8)
        raw_label_array = sitk.GetArrayFromImage(raw_nii)

        # change label
        raw_label_array[raw_label_array != 0] = 1
        # raw_label_array[raw_label_array == 2] = 1

        new_label = sitk.GetImageFromArray(raw_label_array)
        new_label.SetDirection(raw_nii.GetDirection())
        new_label.SetOrigin(raw_nii.GetOrigin())

        sitk.WriteImage(new_label, os.path.join(changed_label_path, nii_file.replace('pancreas', label_name)))

        print(str(cnt) + "finished")
        cnt += 1


if __name__ == '__main__':
    raw_label_path = r'/media/amax4090/MyDisk/zzh/pancreatic segmentation/dataset/MSD/labels'
    new_label_path = r'/media/amax4090/MyDisk/zzh/pancreatic segmentation/dataset/MSD/test_labels/pancreas'
    change_label(raw_label_path, new_label_path, 'pancreas')
