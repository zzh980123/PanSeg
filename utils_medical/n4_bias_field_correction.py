# -*-coding:utf-8-*-
import os
import shutil
import warnings
from nipype.interfaces.ants import N4BiasFieldCorrection

try:
    import SimpleITK as sitk
except ...:
    print('SimpleITK is not here!')


def correct_bias(in_file, out_file, image_type=sitk.sitkFloat64):
    """
    Corrects the bias using ANTs N4BiasFieldCorrection. If this fails, will then attempt to correct bias using SimpleITK
    :param in_file: nii文件的输入路径
    :param out_file: 校正后的文件保存路径名
    :return: 校正后的nii文件全路径名
    """
    # 使用N4BiasFieldCorrection校正MRI图像的偏置场
    correct = N4BiasFieldCorrection()
    correct.inputs.input_image = in_file
    correct.inputs.output_image = out_file
    try:
        done = correct.run()
        return done.outputs.output_image
    except IOError:
        warnings.warn(RuntimeWarning("ANTs N4BIasFieldCorrection could not be found."
                                     "Will try using SimpleITK for bias field correction"
                                     " which will take much longer. To fix this problem, add N4BiasFieldCorrection"
                                     " to your PATH system variable. (example: EXPORT PATH=${PATH}:/path/to/ants/bin)"))
        input_image = sitk.ReadImage(in_file, image_type)
        output_image = sitk.N4BiasFieldCorrection(input_image, input_image > 0)
        sitk.WriteImage(output_image, out_file)
        return os.path.abspath(out_file)


def normalize_image(in_file, out_file, bias_correction=True):
    # bias_correction：是否需要校正
    if bias_correction:
        correct_bias(in_file, out_file)
    else:
        shutil.copy(in_file, out_file)
    return out_file
