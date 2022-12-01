import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    Obtained from voxelmorph
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        if self.grid.device != flow.device:
            self.grid = self.grid.to(flow.device)

        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    Obtained from voxelmorph
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


def gen_flow_transport(shape_):
    dim_ = len(shape_)

    assert dim_ in [1, 2, 3]

    xyz = [
        torch.arange(i)
        for i in shape_
    ]

    grid = torch.meshgrid(xyz)
    id_grid = torch.stack(grid, 0).unsqueeze(0).type(torch.FloatTensor)

    ones = torch.ones_like(id_grid)
    ones.requires_grad = False
    return ones


def trans_t_flow(grid, t: torch.FloatTensor):
    return grid * t


def gen_flow_scale(shape_):
    dim_ = len(shape_)

    assert dim_ in [1, 2, 3]

    xyz = [
        torch.arange(i)
        for i in shape_
    ]

    grid = torch.meshgrid(xyz)

    mid = [
        i // 2 for i in shape_
    ]

    mid = np.expand_dims(np.array(mid, dtype=np.float32), [0] + [i + 2 for i in range(dim_)])

    id_grid = torch.stack(grid, 0).unsqueeze(0).type(torch.FloatTensor)
    mid = torch.from_numpy(mid)

    grid: torch.Tensor = -(id_grid - mid)

    grid.requires_grad = False

    return grid


def trans_s_flow(grid, scale):
    return grid * (1 - 1 / scale)


def get_s(label):
    size = label.shape[2] * label.shape[3]
    b = label.shape[0]
    s = [0.4 * size / torch.count_nonzero(label[i]) for i in range(b)]
    s = torch.stack(s).reshape((b, 1, 1, 1))
    s = torch.sqrt(s)
    s = torch.clip(s, 0.25, 4)
    return s


def get_t(label):
    b = label.shape[0]
    mid = label.shape[-1] // 2

    zx = torch.any(label, dim=-1)
    midx = [torch.where(zx[i]) for i in range(b)]
    middx = [((midx[i][1][0] + midx[i][1][-1]) / 2 - mid) if midx[i][1].__len__() > 0 else torch.tensor(0)
             for i in range(len(midx))]
    midx = torch.stack(middx).reshape(b, 1, 1, 1)

    zy = torch.any(label, dim=-2)
    midy = [torch.where(zy[i]) for i in range(b)]
    middy = [((midy[i][1][0] + midy[i][1][-1]) / 2 - mid) if midy[i][1].__len__() > 0 else torch.tensor(0)
             for i in range(len(midy))]
    midy = torch.stack(middy).reshape(b, 1, 1, 1)

    xy_normal = torch.concat([midx, midy], 1)
    return xy_normal


def get_normal_image(feature, k, M):
    b, c, h, w = feature.shape
    mid = h // 2
    flatten_features = feature.view((b, c, -1))
    _, topk_index = torch.topk(flatten_features[:, 1:, ...], k)
    y = topk_index // w
    x = topk_index - y * w
    mean_y = (torch.mean(y.float(), -1) - mid).reshape(b, 1, 1, 1)
    mean_x = (torch.mean(x.float(), -1) - mid).reshape(b, 1, 1, 1)
    xy_normal = torch.concat([mean_y, mean_x], 1)

    idx1 = torch.arange(b).view(-1, 1, 1)
    idx2 = torch.arange(c).view(1, -1, 1)
    params = flatten_features[idx1, idx2, topk_index]
    params = torch.mean(params, -1).view(b, c, 1, 1)
    s = params[:, 0:1, ...]
    # s_normal = torch.exp2(M * (2 * torch.sigmoid(s) - 1))
    s_normal = M * torch.sigmoid(s) + 1

    return xy_normal, s_normal


if __name__ == '__main__':
    import utils_medical.preproccess as prp
    import utils_medical.plot as plot

    image = prp.read_image("../data/bus.jpg").astype(np.float32)
    # flow = gen_flow_scale(image.shape[:-1], 1)

    ts_img = torch.from_numpy(image)[None, ...].permute(0, 3, 1, 2)
    scale = torch.ones(1, 2, 1, 1).type(torch.FloatTensor) * 2
    t = torch.tensor([200, 200])[None, :, None, None].type(torch.FloatTensor)

    s_flow = trans_s_flow(gen_flow_scale(ts_img.shape[-2:]), scale)
    t_flow = trans_t_flow(gen_flow_transport(ts_img.shape[-2:]), t)

    sp_trans = SpatialTransformer(image.shape[:-1])
    compose_flow = s_flow + sp_trans(t_flow, s_flow)

    ts_new = sp_trans(ts_img, compose_flow)

    np_new = ts_new.permute(0, 2, 3, 1).numpy()
    np_img = ts_img.permute(0, 2, 3, 1).numpy()
    np_flow = compose_flow.permute(0, 2, 3, 1).numpy()

    import matplotlib.pyplot as plt

    plt.imshow(np_new[0, :, ...] * 255)
    plt.show()
    plot.flow([np_flow[0, ::32, ::32, ...]], width=5)
