# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkCommand
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
from vtkmodules.vtkIOLegacy import vtkStructuredPointsReader
from vtkmodules.vtkRenderingCore import (
    vtkColorTransferFunction,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer,
    vtkVolume,
    vtkVolumeProperty,
    vtkInteractorStyle
)
from vtkmodules.vtkRenderingVolume import vtkFixedPointVolumeRayCastMapper
# noinspection PyUnresolvedReferences
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkOpenGLRayCastImageDisplayHelper, vtkOpenGLGPUVolumeRayCastMapper
from utils_medical.vtk_render import transform

# !/usr/bin/env python

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from utils_medical.vtk_render.transform import *

# !/usr/bin/env python

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
from vtkmodules.vtkRenderingCore import (
    vtkColorTransferFunction,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer,
    vtkVolume,
    vtkVolumeProperty
)
from vtkmodules.vtkRenderingVolume import vtkFixedPointVolumeRayCastMapper
# noinspection PyUnresolvedReferences
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkOpenGLRayCastImageDisplayHelper


def point_color_map(colors, max_=None, min_=None):
    max_ = max(colors) if not max_ else max_
    min_ = min(colors) if not min_ else min_

    #  n bit tp 24 bit rgb color
    #  long to rgb
    target_max = 0xffff
    target_min = 0x0000

    r_mask = 0x000f
    g_mask = 0x00f0
    b_mask = 0x0f00

    trans_fun = lambda color_: int((color_ - min_) / (max_ - min_) * (target_max - target_min))

    color_map = dict()
    for color in colors:
        color2 = trans_fun(color)
        r = color2 & r_mask
        g = color2 & g_mask
        b = color2 & b_mask
        color_map[color] = (r, g, b)

    return color_map


def auto_vtk_color_trans_fun(colors, max_=None, min_=None):
    color_map = point_color_map(colors, max_, min_)
    color_transfer_function = vtkColorTransferFunction()

    for k, v in color_map.items():
        color_transfer_function.AddRGBPoint(k, *v)

    return color_transfer_function


def render_window(window_size=(900, 900), background_color=(.5, .5, .5)):
    ren = vtkRenderer()
    # ren.AddVolume(volume)
    ren.SetBackground(*background_color)

    renWin = vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(*window_size)

    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renWin)
    # style = vtkInteractorStyle()
    # style.SetTDxStyle(vtkmodules.vtkInteractionStyle.VTKIS_TRACKBALL)
    interactor.SetInteractorStyle(vtkmodules.vtkInteractionStyle.vtkInteractorStyleTrackballCamera())
    interactor.Initialize()
    interactor.UserCallback()
    # renWin.Render()

    # camera = ren.GetActiveCamera()
    # camera.SetPosition(42.301174, 939.893457, -124.005030)
    # camera.SetFocalPoint(224.697134, 221.301653, 146.823706)
    # camera.SetViewUp(0.262286, -0.281321, -0.923073)
    # camera.SetDistance(789.297581)
    # camera.SetClippingRange(168.744328, 1509.660206)

    # interactor.Start()
    # camera = ren.GetActiveCamera()
    # c = volume.GetCenter()
    # camera.SetViewUp(0, 0, -1)
    # camera.SetPosition(c[0], c[1] - 400, c[2])
    # camera.SetFocalPoint(c[0], c[1], c[2])
    # camera.Azimuth(30.0)
    # camera.Elevation(30.0)
    # interactor.Start()

    return renWin, ren, interactor


def volume_create4vtk(np_data: np.ndarray, setup_volume_property_fun=None, color_transfer_function=None,
                      opacity_transfer_function=None, volume_gradient_opacity=None):
    np_data = transform.numpy2vtk(np_data)

    if not color_transfer_function:
        color_transfer_function = vtkColorTransferFunction()
        color_transfer_function.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
        color_transfer_function.AddRGBPoint(64.0, 1.0, 0.0, 0.0)
        color_transfer_function.AddRGBPoint(128.0, 0.0, 0.0, 1.0)
        color_transfer_function.AddRGBPoint(192.0, 0.0, 1.0, 0.0)
        color_transfer_function.AddRGBPoint(255.0, 0.0, 0.2, 0.0)

    if not opacity_transfer_function:
        opacity_transfer_function = vtkPiecewiseFunction()
        opacity_transfer_function.AddPoint(0, 0)
        opacity_transfer_function.AddPoint(1, 1)

    if not volume_gradient_opacity:
        volume_gradient_opacity = vtkPiecewiseFunction()
        volume_gradient_opacity.AddPoint(0, 0)
        volume_gradient_opacity.AddPoint(1, 1)
        # volume_gradient_opacity.AddPoint(255, 1.0)

    volume_property = vtkVolumeProperty()
    volume_property.SetColor(color_transfer_function)
    volume_property.SetScalarOpacity(opacity_transfer_function)
    volume_property.SetGradientOpacity(volume_gradient_opacity)
    # volume_property.SetIndependentComponents(2)
    volume_property.ShadeOn()
    volume_property.SetInterpolationTypeToLinear()
    volume_property.SetAmbient(0.4)
    volume_property.SetDiffuse(0.6)
    volume_property.SetSpecular(0.2)

    if setup_volume_property_fun:
        setup_volume_property_fun(volume_property)

    volume_mapper = vtkFixedPointVolumeRayCastMapper()
    volume_mapper.SetInputData(np_data)
    # volume_mapper.SetNumberOfThreads(10)
    # volume_mapper.SetBlendModeToMaximumIntensity()

    volume = vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)
    return volume


def window_add_vol(render, vol):
    render.AddVolume(vol)
    # render.Render()


def window_rm_vol(render, vol):
    render.RemoveVolume(vol)


def window_get_vols(render_window):
    return render_window.GetVolumes()


def volume_render(np_data: np.ndarray, window_size=(900, 900), setup_volume_property_fun=None, background_color=(.5, .5, .5), color_transfer_function=None,
                  opacity_transfer_function=None, volume_gradient_opacity=None):
    np_data = transform.numpy2vtk(np_data)

    if not color_transfer_function:
        color_transfer_function = vtkColorTransferFunction()
        color_transfer_function.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
        color_transfer_function.AddRGBPoint(64.0, 1.0, 0.0, 0.0)
        color_transfer_function.AddRGBPoint(128.0, 1.0, 0.0, 1.0)
        color_transfer_function.AddRGBPoint(192.0, 0.0, 1.0, 1.0)
        color_transfer_function.AddRGBPoint(255.0, 0.0, 1.0, 0.0)

    if not opacity_transfer_function:
        opacity_transfer_function = vtkPiecewiseFunction()
        opacity_transfer_function.AddPoint(0, 0)
        opacity_transfer_function.AddPoint(1, 1)

    if not volume_gradient_opacity:
        volume_gradient_opacity = vtkPiecewiseFunction()
        volume_gradient_opacity.AddPoint(0, 0)
        volume_gradient_opacity.AddPoint(1, 1)
        # volume_gradient_opacity.AddPoint(255, 1.0)

    volume_property = vtkVolumeProperty()
    volume_property.SetColor(color_transfer_function)
    volume_property.SetScalarOpacity(opacity_transfer_function)
    volume_property.SetGradientOpacity(volume_gradient_opacity)
    # volume_property.SetIndependentComponents(2)
    # volume_property.ShadeOn()
    volume_property.SetInterpolationTypeToLinear()
    volume_property.SetAmbient(0.4)
    volume_property.SetDiffuse(0.6)
    volume_property.SetSpecular(0.2)

    if setup_volume_property_fun:
        setup_volume_property_fun(volume_property)

    volume_mapper = vtkFixedPointVolumeRayCastMapper()
    volume_mapper.SetInputData(np_data)
    # volume_mapper.SetNumberOfThreads(10)
    # volume_mapper.SetBlendModeToMaximumIntensity()

    volume = vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    ren = vtkRenderer()
    ren.AddVolume(volume)
    ren.SetBackground(*background_color)

    renWin = vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(*window_size)

    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renWin)
    interactor.SetInteractorStyle(vtkmodules.vtkInteractionStyle.vtkInteractorStyleTrackballCamera())

    interactor.Initialize()
    renWin.Render()

    # renWin.Render()
    #
    # camera = ren.GetActiveCamera()
    # camera.SetPosition(42.301174, 939.893457, -124.005030)
    # camera.SetFocalPoint(224.697134, 221.301653, 146.823706)
    # camera.SetViewUp(0.262286, -0.281321, -0.923073)
    # camera.SetDistance(789.297581)
    # camera.SetClippingRange(168.744328, 1509.660206)
    #
    # interactor.Start()
    # camera = ren.GetActiveCamera()
    # c = volume.GetCenter()
    # camera.SetViewUp(0, 0, -1)
    # camera.SetPosition(c[0], c[1] - 400, c[2])
    # camera.SetFocalPoint(c[0], c[1], c[2])
    # camera.Azimuth(30.0)
    # camera.Elevation(30.0)
    interactor.Start()


from vtkmodules.vtkFiltersCore import (vtkFlyingEdges3D, vtkMarchingCubes)
from vtkmodules.vtkFiltersGeneral import vtkDiscreteMarchingCubes
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor
from vtkmodules.vtkFiltersCore import vtkWindowedSincPolyDataFilter

def surface_render(np_data: np.ndarray, window_size=(900, 900), ):
    # color_transfer_function = vtkColorTransferFunction()
    # color_transfer_function.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
    # color_transfer_function.AddRGBPoint(64.0, 1.0, 0.0, 0.0)
    # color_transfer_function.AddRGBPoint(128.0, 1.0, 0.0, 1.0)
    # color_transfer_function.AddRGBPoint(192.0, 0.0, 1.0, 1.0)
    # color_transfer_function.AddRGBPoint(255.0, 0.0, 1.0, 0.0)

    se = vtkDiscreteMarchingCubes()
    vtk_image = transform.numpy2vtk(np_data)
    se.SetInputData(vtk_image)

    smoother = vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(se.GetOutputPort())
    smoother.SetNumberOfIterations(10)
    # smoother.BoundarySmoothingOff()
    # smoother.FeatureEdgeSmoothingOff()
    smoother.SetPassBand(0.01)
    smoother.SetFeatureAngle(120)
    smoother.NormalizeCoordinatesOn()
    smoother.NonManifoldSmoothingOn()
    smoother.Update()

    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(smoother.GetOutputPort())

    ren = vtkRenderer()
    ren_win = vtkRenderWindow()
    ren_win.AddRenderer(ren)

    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(ren_win)
    interactor.SetInteractorStyle(vtkmodules.vtkInteractionStyle.vtkInteractorStyleTrackballCamera())
    interactor.SetRenderWindow(ren_win)

    actor = vtkActor()
    actor.SetMapper(mapper)

    ren.AddActor(actor)

    ren_win.Render()

    interactor.Start()


class VtkVolumeScene:

    def __init__(self):
        self.run = False

    def start(self):
        self.run = True


class MyEvent(vtkCommand):

    def Execute(self, caller, eventId, callData):  # real signature unknown; restored from __doc__
        """
        Execute(self, caller:vtkObject, eventId:int, callData:Pointer)
            -> None
        C++: virtual void Execute(vtkObject *caller,
            unsigned long eventId, void *callData)

        All derived classes of vtkCommand must implement this method.
        This is the method that actually does the work of the callback.
        The caller argument is the object invoking the event, the eventId
        parameter is the id of the event, and callData parameter is data
        that can be passed into the execute method. (Note:
        vtkObject::InvokeEvent() takes two parameters: the event id (or
        name) and call data. Typically call data is nullptr, but the user
        can package data and pass it this way. Alternatively, a derived
        class of vtkCommand can be used to pass data.)
        """
        renw = caller.GetRenderWinow()
        renderer = renw.GetRenderer()


if __name__ == '__main__':
    import numpy as np
    import utils_medical.preproccess as prp

    a = prp.read_one_voxel('/media/kevin/870A38D039F26F71/PycharmProjects/MRRNet/datasets/MLT/cropped/labels/0013.nii.gz')
    # b = np.load('/media/kevin/870A38D039F26F71/PycharmProjects/MRRNet/datasets/MLT/cropped/augmentation/labels/trans_0_0015.nii.gz.landmark.npy')
    #
    # volume_render(a)
    rw, r, i = render_window()
    window_add_vol(r, volume_create4vtk(a))
    # rw.RemoveRenderer(r)
    # rw.AddRenderer(r)
    rw.Render()
    i.CreateRepeatingTimer(10)
    i.AddObserver(vtkCommand.TimerEvent, MyEvent)
    i.Start()

    res = window_get_vols(r)
    print(res)
