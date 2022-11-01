import numpy as np
from vtkmodules.util import numpy_support, vtkConstants
from vtkmodules.vtkCommonDataModel import *


def vtk2numpy(data):
    temp = numpy_support.vtk_to_numpy(data.GetPointData().GetScalars())
    dims = data.GetDimensions()
    component = data.GetNumberOfScalarComponents()
    if component == 1:
        numpy_data = temp.reshape(dims[2], dims[1], dims[0])
        numpy_data = numpy_data.transpose(2, 1, 0)
    elif component == 3 or component == 4:
        if dims[2] == 1:  # a 2D RGB image
            numpy_data = temp.reshape(dims[1], dims[0], component)
            numpy_data = numpy_data.transpose(0, 1, 2)
            numpy_data = np.flipud(numpy_data)
        else:
            raise RuntimeError('unknow type')
    else:
        raise RuntimeError('only support 1, 3, 4 components')
    return numpy_data


def numpy2vtk(data, multi_component=False, vtk_type='float') -> vtkImageData:
    """
    multi_components: rgb has 3 components
    type：float or char
    """
    if vtk_type == 'float':
        data_type = vtkConstants.VTK_FLOAT
    elif vtk_type == 'char':
        data_type = vtkConstants.VTK_UNSIGNED_CHAR
    else:
        raise RuntimeError('unknown type')
    if not multi_component:
        if len(data.shape) == 2:
            data = data[:, :, np.newaxis]
        flat_data_array = data.transpose(2, 1, 0).flatten()
        vtk_data = numpy_support.numpy_to_vtk(num_array=flat_data_array, deep=True, array_type=data_type)
        shape = data.shape
    else:
        assert len(data.shape) == 3, 'only test for 2D RGB'
        flat_data_array = data.transpose(1, 0, 2)
        flat_data_array = np.reshape(flat_data_array, newshape=[-1, data.shape[2]])
        vtk_data = numpy_support.numpy_to_vtk(num_array=flat_data_array, deep=True, array_type=data_type)
        shape = [data.shape[0], data.shape[1], 1]
    img = vtkImageData()
    img.GetPointData().SetScalars(vtk_data)
    img.SetDimensions(shape[0], shape[1], shape[2])
    return img
