"""
tensorflow/keras plot utilities for the neuron project

If you use this code, please cite 
Dalca AV, Guttag J, Sabuncu MR
Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
CVPR 2018

Contact: adalca [at] csail [dot] mit [dot] edu
License: GPLv3
"""

# third party
from logging import warning

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable  # plotting
import matplotlib

matplotlib.use('agg')


def get_matplotlib_version():
    try:
        import matplotlib
    except Exception as e:
        return "-1"

    try:
        version = matplotlib._version.version
    except Exception as e:
        version = matplotlib._version.get_versions()['version']

    return version, matplotlib.get_backend()


def memory_lack_warning():
    version_list = ["3.4.2", "3.4.0", "3.4.1", "3.5.1", "3.5.0", ]
    v, b = get_matplotlib_version()
    if v in version_list and b != 'agg':
        warning("The matplotlib's figure has memory lack issues if use none ‘agg’ backend, do not use it at a long-time loop block!")


def _recheck():
    def plot(dat_, i_):
        fig = plt.figure(figsize=(128, 128), clear=True)
        plt.close('all')

    for i in range(20):
        dat = np.random.rand(128, 128)

        plot(dat, i)
        del dat


def check_memoryleak(function_):
    import tracemalloc, gc
    tracemalloc.start()
    current, peak = tracemalloc.get_traced_memory()
    print("Before: memory usage is {}MB, peak was {}".format(current / 1e6, peak / 1e6))
    function_()
    current, peak = tracemalloc.get_traced_memory()
    print("After(GC Prepared): memory usage is {}MB, peak was {}".format(current / 1e6, peak / 1e6))
    gc.collect()
    after, peak = tracemalloc.get_traced_memory()
    print("After(GC Done): memory usage is {}MB, peak was {}".format(after / 1e6, peak / 1e6))
    tracemalloc.stop()
    if (current - after - 0.09) / after < 0.1:
        warning("This function may be memory leak")


memory_lack_warning()
# check_memoryleak(_recheck)

"""
The above codes are to detect the memory lack in matploatlib.
May be remove when the version_list's version is out of date.
by nowandfuture. 

"""


def slices(slices_in,  # the 2D slices
           titles=None,  # list of titles
           cmaps=None,  # list of colormaps
           norms=None,  # list of normalizations
           do_colorbars=False,  # option to show colorbars on each slice
           grid=False,  # option to plot the images in a grid or a single row
           width=15,  # width in in
           show=True,  # option to actually show the plot (plt.show())
           imshow_args=None):
    '''
    plot a grid of slices (2d images)
    '''

    # input processing
    nb_plots = len(slices_in)
    for si, slice_in in enumerate(slices_in):
        assert len(slice_in.shape) == 2, 'each slice has to be 2d: 2d channels'
        slices_in[si] = slice_in.astype('float')

    def input_check(inputs, nb_plots, name):
        ''' change input from None/single-link '''
        assert (inputs is None) or (len(inputs) == nb_plots) or (len(inputs) == 1), \
            'number of %s is incorrect' % name
        if inputs is None:
            inputs = [None]
        if len(inputs) == 1:
            inputs = [inputs[0] for i in range(nb_plots)]
        return inputs

    titles = input_check(titles, nb_plots, 'titles')
    cmaps = input_check(cmaps, nb_plots, 'cmaps')
    norms = input_check(norms, nb_plots, 'norms')
    imshow_args = input_check(imshow_args, nb_plots, 'imshow_args')
    for idx, ia in enumerate(imshow_args):
        imshow_args[idx] = {} if ia is None else ia

    # figure out the number of rows and columns
    if grid:
        if isinstance(grid, bool):
            rows = np.floor(np.sqrt(nb_plots)).astype(int)
            cols = np.ceil(nb_plots / rows).astype(int)
        else:
            assert isinstance(grid, (list, tuple)), \
                "grid should either be bool or [rows,cols]"
            rows, cols = grid
    else:
        rows = 1
        cols = nb_plots

    # prepare the subplot
    fig, axs = plt.subplots(rows, cols)
    if rows == 1 and cols == 1:
        axs = [axs]

    for i in range(nb_plots):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        # turn off axis
        ax.axis('off')

        # add titles
        if titles is not None and titles[i] is not None:
            ax.title.set_text(titles[i])

        # show figure
        im_ax = ax.imshow(slices_in[i], cmap=cmaps[i], interpolation="nearest", norm=norms[i], **imshow_args[i])

        # colorbars
        # http://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
        if do_colorbars and cmaps[i] is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im_ax, cax=cax)

    # clear axes that are unnecessary
    for i in range(nb_plots, col * row):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        ax.axis('off')

    # show the plots
    fig.set_size_inches(width, rows / cols * width)
    plt.tight_layout()

    if show:
        plt.show()

    return (fig, axs, plt)


def flow_legend():
    """
    show quiver plot to indicate how arrows are colored in the flow() method.
    https://stackoverflow.com/questions/40026718/different-colours-for-arrows-in-quiver-plot
    """
    ph = np.linspace(0, 2 * np.pi, 13)
    x = np.cos(ph)
    y = np.sin(ph)
    u = np.cos(ph)
    v = np.sin(ph)
    colors = np.arctan2(u, v)

    norm = Normalize()
    norm.autoscale(colors)
    # we need to normalize our colors array to match it colormap domain
    # which is [0, 1]

    colormap = cm.winter

    plt.figure(figsize=(6, 6))
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.quiver(x, y, u, v, color=colormap(norm(colors)), angles='xy', scale_units='xy', scale=1)
    plt.show()


def flow(slices_in,  # the 2D slices
         titles=None,  # list of titles
         cmaps=None,  # list of colormaps
         width=15,  # width in in
         img_indexing=True,  # whether to match the image view, i.e. flip y axis
         grid=False,  # option to plot the images in a grid or a single row
         show=True,  # option to actually show the plot (plt.show())
         scale=1):  # note quiver essentially draws quiver length = 1/scale
    '''
    plot a grid of flows (2d+2 images)
    '''

    # input processing
    nb_plots = len(slices_in)
    for slice_in in slices_in:
        assert len(slice_in.shape) == 3, 'each slice has to be 3d: 2d+2 channels'
        assert slice_in.shape[-1] == 2, 'each slice has to be 3d: 2d+2 channels'

    def input_check(inputs, nb_plots, name):
        ''' change input from None/single-link '''
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        assert (inputs is None) or (len(inputs) == nb_plots) or (len(inputs) == 1), \
            'number of %s is incorrect' % name
        if inputs is None:
            inputs = [None]
        if len(inputs) == 1:
            inputs = [inputs[0] for _ in range(nb_plots)]
        return inputs

    if img_indexing:
        for si, slc in enumerate(slices_in):
            slices_in[si] = np.flipud(slc)

    titles = input_check(titles, nb_plots, 'titles')
    cmaps = input_check(cmaps, nb_plots, 'cmaps')
    scale = input_check(scale, nb_plots, 'scale')

    # figure out the number of rows and columns
    if grid:
        if isinstance(grid, bool):
            rows = np.floor(np.sqrt(nb_plots)).astype(int)
            cols = np.ceil(nb_plots / rows).astype(int)
        else:
            assert isinstance(grid, (list, tuple)), \
                "grid should either be bool or [rows,cols]"
            rows, cols = grid
    else:
        rows = 1
        cols = nb_plots

    # prepare the subplot
    fig, axs = plt.subplots(rows, cols)
    if rows == 1 and cols == 1:
        axs = [axs]

    for i in range(nb_plots):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        # turn off axis
        ax.axis('off')

        # add titles
        if titles is not None and titles[i] is not None:
            ax.title.set_text(titles[i])

        u, v = slices_in[i][..., 0], slices_in[i][..., 1]
        colors = np.arctan2(u, v)
        colors[np.isnan(colors)] = 0
        norm = Normalize()
        norm.autoscale(colors)
        if cmaps[i] is None:
            colormap = cm.winter
        else:
            raise Exception("custom cmaps not currently implemented for plt.flow()")

        # show figure
        ax.quiver(u, v,
                  color=colormap(norm(colors).flatten()),
                  angles='xy',
                  units='xy',
                  scale=scale[i])
        ax.axis('equal')

    # clear axes that are unnecessary
    for i in range(nb_plots, col * row):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        ax.axis('off')

    # show the plots
    fig.set_size_inches(width, rows / cols * width)
    plt.tight_layout()

    if show:
        plt.show()
        plt.close(fig)

    return fig, axs, plt


def fig2data(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    import PIL.Image as Image
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    return image


def plot_3d(image, threshold=-300):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)
    import skimage.measure as measure

    verts, faces = measure.marching_cubes_classic(p, threshold)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    mesh = Poly3DCollection(verts[faces], alpha=0.5)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.show()

    plt.close()


def get_jac_tf(displacement):
    '''
    the expected input: displacement of shape(batch, H, W, D, channel),
    obtained in TensorFlow.
    '''
    D_y = (displacement[:, 1:, :-1, :-1, :] - displacement[:, :-1, :-1, :-1, :])
    D_x = (displacement[:, :-1, 1:, :-1, :] - displacement[:, :-1, :-1, :-1, :])
    D_z = (displacement[:, :-1, :-1, 1:, :] - displacement[:, :-1, :-1, :-1, :])

    D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_y[..., 2] * D_z[..., 1])
    D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_z[..., 0])
    D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])

    D = D1 - D2 + D3

    return D


def get_jac_pt(displacement):
    '''
    the expected input: displacement of shape(batch, channel, H, W, D),
    obtained in TensorFlow.
    '''
    D_y = (displacement[:, :, 1:, :-1, :-1] - displacement[:, :, :-1, :-1, :-1])
    D_x = (displacement[:, :, :-1, 1:, :-1] - displacement[:, :, :-1, :-1, :-1])
    D_z = (displacement[:, :, :-1, :-1, 1:] - displacement[:, :, :-1, :-1, :-1])

    D1 = (D_x[:, 0, ...] + 1) * ((D_y[:, 1, ...] + 1) * (D_z[:, 2, ...] + 1) - D_y[:, 2, ...] * D_z[:, 1, ...])
    D2 = (D_x[:, 1, ...]) * (D_y[:, 0, ...] * (D_z[:, 2, ...] + 1) - D_y[:, 2, ...] * D_z[:, 0, ...])
    D3 = (D_x[:, 2, ...]) * (D_y[:, 0, ...] * D_z[:, 1, ...] - (D_y[:, 1, ...] + 1) * D_z[:, 0, ...])

    D = D1 - D2 + D3

    return D


def get_jac_np(displacement):
    '''
    the expected input: displacement of shape(channel, H, W, D),
    obtained in TensorFlow.
    '''
    D_y = (displacement[:, 1:, :-1, :-1] - displacement[:, :-1, :-1, :-1])
    D_x = (displacement[:, :-1, 1:, :-1] - displacement[:, :-1, :-1, :-1])
    D_z = (displacement[:, :-1, :-1, 1:] - displacement[:, :-1, :-1, :-1])

    D1 = (D_x[0, ...] + 1) * ((D_y[1, ...] + 1) * (D_z[2, ...] + 1) - D_y[2, ...] * D_z[1, ...])
    D2 = (D_x[1, ...]) * (D_y[0, ...] * (D_z[2, ...] + 1) - D_y[2, ...] * D_z[0, ...])
    D3 = (D_x[2, ...]) * (D_y[0, ...] * D_z[1, ...] - (D_y[1, ...] + 1) * D_z[0, ...])

    D = D1 - D2 + D3

    return D

############################ This code is from FAIM ###################################

"""
Created on Wed Apr 11 10:08:36 2018
@author: Dongyang
This script contains some utilize functions for data visualization
"""
from matplotlib import colors


# ==============================================================================
# Define a custom colormap for visualiza Jacobian
# ==============================================================================
class JacColorNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, zero_value=None, one_value=None, clip=False):
        self.zero_value = zero_value
        self.one_value = one_value
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        res = np.sum(value < 0) / (value.shape[0] * value.shape[1])
        print(res, np.sum(value <= 0))

        value[value < 0] -= 1
        value += 1

        x, y = [self.vmin - 1, self.zero_value - 1, self.one_value + 1, self.vmax + 1], [1, 0.8, 0.47, 0]
        res = np.ma.masked_array(np.interp(value, x, y))
        return res
