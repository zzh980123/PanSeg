import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import utils_medical.pytorch.flow as flow
import models.common as common
import math


# class GridManager(nn.Module):
#
#     def __init__(self, chn_in, chn_out, shape_=None):
#         super().__init__()
#         GridManager.instance = self


class ScaleConvLayer(nn.Module):

    def __init__(self, chn_in, chn_out, shape_=None, max_scaling=4):
        super().__init__()

        self.max_scaling = max_scaling
        self.min_scaling = 1 / max_scaling
        assert self.max_scaling > 0
        self.M = math.log2(self.max_scaling)

        self.conv1 = common.GhostBottleneck(chn_in, chn_in)
        self.conv2 = common.GhostBottleneck(chn_in, chn_in)
        self.conv2_ex_1 = nn.Conv2d(in_channels=chn_in, out_channels=chn_out, kernel_size=3, padding=1)
        self.normal1 = nn.BatchNorm2d(chn_out)

        self.reg = nn.Conv2d(in_channels=chn_out, out_channels=4, kernel_size=1, padding=0)
        self.max_pooling = nn.AdaptiveMaxPool2d(1, return_indices=True)

        # self.pooling = nn.AdaptiveAvgPool2d(1)

        self.flatten_pt = nn.Conv2d(in_channels=chn_in, out_channels=4, kernel_size=1, padding=0)

        self.conv3 = nn.Conv2d(in_channels=4, out_channels=chn_out, kernel_size=3, padding=1)
        self.conv4 = common.GhostBottleneck(chn_out, chn_out)

        self.normal2 = nn.BatchNorm2d(chn_out)

        # self.s_flow = self.t_flow = None
        # self.trans = None

        self.dyn_flow_shape = True

        if shape_ is not None:
            self.register_buffer('s_flow', nn.Parameter(flow.gen_flow_scale(shape_), requires_grad=False))
            self.register_buffer('t_flow', nn.Parameter(flow.gen_flow_transport(shape_), requires_grad=False))
            self.trans = flow.SpatialTransformer(shape_, mode='nearest')
            self.dyn_flow_shape = False

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def get_st_flows(self, shape_, device):

        if self.dyn_flow_shape:
            self.register_buffer('s_flow', nn.Parameter(flow.gen_flow_scale(shape_[2:]), requires_grad=False))
            self.register_buffer('t_flow', nn.Parameter(flow.gen_flow_transport(shape_[2:]), requires_grad=False))
            self.trans = flow.SpatialTransformer(shape_[2:], mode='nearest')

            self.s_flow = self.s_flow.to(device)
            self.t_flow = self.t_flow.to(device)
            self.dyn_flow_shape = False

        return self.s_flow, self.t_flow

    def forward(self, x, origin=None):
        shape_ = x.shape
        short_cut = origin

        h = self.conv2(self.conv1(x))
        h = F.leaky_relu(h)
        # x = self.conv2(h)
        # x = F.leaky_relu(x)

        x = self.conv2_ex_1(x)
        x = self.normal1(x)

        params = self.reg(x)

        B, C, H, W = params.shape
        _, max_index = self.max_pooling(params[:, 3:, ...])
        max_index = torch.repeat_interleave(max_index, C, dim=1).view((B, C, 1))
        flat_params = params.view((B, C, H * W, 1))
        idx1 = torch.arange(B).view(-1, 1, 1)  # [B, 1, 1]
        idx2 = torch.arange(C).view(1, -1, 1)  # [1, C, 1]
        params = flat_params[idx1, idx2, max_index]

        xy, s, c = params[:, 0:2, ...], params[:, 2:3, ...], params[:, 3:, ...]

        # c_normal = torch.sigmoid(c)
        xy_normal = torch.sigmoid(xy) - 0.5
        s_normal = torch.exp2(self.M * (2 * torch.sigmoid(s) - 1))

        C = xy_normal.shape[1]

        for i in range(C):
            xy_normal[:, i, ...] *= shape_[i + 2]

        s_flow, t_flow = self.get_st_flows(shape_, x.device)

        s_flow = flow.trans_s_flow(s_flow, s_normal)
        t_flow = flow.trans_t_flow(t_flow, xy_normal)

        flatten_ps = self.flatten_pt(h)

        if short_cut is not None:
            flatten_ps = short_cut + flatten_ps

        trans_fps = self.trans(self.trans(flatten_ps, t_flow), s_flow)
        trans_fps = self.conv3(trans_fps)
        trans_fps = self.normal2(self.conv4(trans_fps))

        return trans_fps, params


class ScaleBackConvLayer(nn.Module):

    def __init__(self, chn_in, chn_out, shape_=None, max_scaling=4):
        super().__init__()
        self.max_scaling = max_scaling
        self.min_scaling = 1 / max_scaling
        assert self.max_scaling > 0
        self.M = math.log2(self.max_scaling)
        self.flatten_pt = nn.Conv2d(in_channels=chn_in, out_channels=1, kernel_size=1, padding=0)

        self.conv3 = nn.Conv2d(in_channels=1, out_channels=chn_out, kernel_size=3, padding=1)
        self.conv4 = common.GhostBottleneck(chn_out, chn_out)

        self.dyn_flow_shape = True
        if shape_ is not None:
            self.register_buffer('s_flow', nn.Parameter(flow.gen_flow_scale(shape_[2:]), requires_grad=False))
            self.register_buffer('t_flow', nn.Parameter(flow.gen_flow_transport(shape_[2:]), requires_grad=False))
            self.trans = flow.SpatialTransformer(shape_, mode='nearest')
            self.dyn_flow_shape = False

    # def get_st_flows(self, shape_, device):
    #
    #     if self.dyn_flow_shape:
    #         self.register_buffer('s_flow', nn.Parameter(flow.gen_flow_scale(shape_[2:]), requires_grad=False))
    #         self.register_buffer('t_flow', nn.Parameter(flow.gen_flow_transport(shape_[2:]), requires_grad=False))
    #         self.s_flow = self.s_flow.to(device).repeat_interleave(shape_[0], 0)
    #         self.t_flow = self.t_flow.to(device).repeat_interleave(shape_[0], 0)
    #         self.dyn_flow_shape = False
    #
    #     return self.s_flow, self.t_flow

    def get_st_flows(self, shape_, device):

        if self.dyn_flow_shape:
            self.register_buffer('s_flow', nn.Parameter(flow.gen_flow_scale(shape_[2:]), requires_grad=False))
            self.register_buffer('t_flow', nn.Parameter(flow.gen_flow_transport(shape_[2:]), requires_grad=False))
            self.trans = flow.SpatialTransformer(shape_[2:], mode='nearest')

            self.s_flow = self.s_flow.to(device)
            self.t_flow = self.t_flow.to(device)
            self.dyn_flow_shape = False

        return self.s_flow, self.t_flow

    def forward(self, x, params):
        shape_ = x.shape

        xy, s, c = params[:, 0:2, ...], params[:, 2:3, ...], params[:, 3:4, ...]

        xy_normal = torch.sigmoid(xy) - 0.5
        s_normal = torch.exp2(self.M * (2 * torch.sigmoid(s) - 1))

        C = xy_normal.shape[1]

        for i in range(C):
            xy_normal[:, i, ...] *= shape_[i + 2]

        s_flow, t_flow = self.get_st_flows(shape_, x.device)

        s_flow = flow.trans_s_flow(s_flow, 1 / s_normal)
        t_flow = flow.trans_t_flow(t_flow, -1 * xy_normal)

        flatten_ps = self.flatten_pt(x)

        trans_back = self.trans(self.trans(flatten_ps, s_flow), t_flow)
        trans_fps = self.conv3(trans_back)
        trans_fps = self.conv4(trans_fps)

        return trans_fps, trans_back


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the field *and* rescaling it.
    """

    def __init__(self, resize, ndims):
        super().__init__()
        self.factor = 1.0 / resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x


class Conv3x3Block(nn.Module):
    def __init__(self, in_channels, out_channels, dim=2, normalize=False):
        super().__init__()
        self.conv = conv3x3(in_channels=in_channels, out_channels=out_channels, dim=dim)
        self.act = nn.LeakyReLU()
        normal_nd = getattr(nn, f"BatchNorm{dim}d")
        self.normal = None
        if normalize:
            self.normal = normal_nd(out_channels)

    def forward(self, x):
        res = self.act(self.conv(x))
        if self.normal:
            res = self.normal(res)

        return res


class Conv3x3BlockTail(nn.Module):
    def __init__(self, in_channels, out_channels, dim=2, normalize=False):
        super().__init__()
        self.conv = conv3x3(in_channels=in_channels, out_channels=out_channels, dim=dim)
        self.act = nn.LeakyReLU()
        normal_nd = getattr(nn, f"BatchNorm{dim}d")
        self.normal = None
        if normalize:
            self.normal = normal_nd(out_channels)

    def forward(self, x):
        res = self.conv(x)
        if self.normal:
            res = self.normal(res)

        return self.act(res)


def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1, dim=3):
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups) if dim == 3 else \
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups)


def upconv2x2(in_channels, out_channels, mode='transpose', dim=3):
    if mode == 'transpose':
        return nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2) if dim == 3 else \
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            ResizeTransform(1 / 2, ndims=dim),
            conv1x1(in_channels, out_channels, dim=dim))


def conv1x1(in_channels, out_channels, groups=1, dim=3):
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1) if dim == 3 \
        else nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


def bn(out_channels, dim=3):
    return nn.BatchNorm2d(out_channels) if dim == 2 else nn.BatchNorm3d(out_channels)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, pooling=True, dim=3):
        super(DownConv, self).__init__()
        assert dim in [2, 3]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels, dim=dim)
        self.conv2 = conv3x3(self.out_channels, self.out_channels, dim=dim)
        self.bn1 = bn(out_channels, dim=dim)
        self.bn2 = bn(out_channels, dim=dim)

        if self.pooling:
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2) if dim == 3 else nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels,
                 merge_mode='concat', up_mode='transpose', dim=3):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.bn1 = bn(out_channels, dim=dim)
        self.bn2 = bn(out_channels, dim=dim)
        self.upconv = upconv2x2(self.in_channels, self.out_channels,
                                mode=self.up_mode, dim=dim)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(
                2 * self.out_channels, self.out_channels, dim=dim)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels, dim=dim)
        self.conv2 = conv3x3(self.out_channels, self.out_channels, dim=dim)

    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down

        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        return x


class UNet(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597

    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).

    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, num_classes, in_channels=3, depth=5,
                 start_filts=64, up_mode='transpose',
                 merge_mode='concat', dim=3):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        global outs
        super(UNet, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2 ** i)
            pooling = True if i < depth - 1 else False

            down_conv = DownConv(ins, outs, pooling=pooling, dim=dim)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                             merge_mode=merge_mode, dim=dim)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.num_classes, dim=dim)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        # x = torch.sigmoid(x)
        # x = torch.softmax(x, 1)

        return x


class ScaleSegmentNet(nn.Module):

    def __init__(self, in_channel=3, branch_num=16, classes_num=2, feature_num=[3, 3, 9, 3], dim=2):
        super().__init__()
        self.branch_num = branch_num
        unit = 4
        self.unit = unit
        self.head = Conv3x3Block(in_channels=in_channel, out_channels=unit, dim=dim)
        self.branch_spilt = Conv3x3Block(in_channels=unit, out_channels=unit * branch_num, dim=dim, normalize=True)

        self.branch_forward_layers = nn.ModuleList(
            [
                ScaleConvLayer(unit, unit * feature_num[0])
                for _ in range(branch_num)
            ]
        )

        # self.extract_layers = nn.ModuleList(
        #     [
        #         DownConv(feature_num[0] * unit, feature_num[1] * unit, dim=dim),
        #         DownConv(feature_num[1] * unit, feature_num[2] * unit, dim=dim),
        #         # Conv3x3Block(in_channels=feature_num[0] * unit, out_channels=feature_num[1] * unit, dim=dim, normalize=True),
        #         # Conv3x3Block(in_channels=feature_num[1] * unit, out_channels=unit, dim=dim, normalize=True),
        #         UpConv(feature_num[2] * unit, out_channels=unit * feature_num[1], dim=dim),
        #         UpConv(feature_num[1] * unit, out_channels=unit, dim=dim)
        #     ]
        # )

        self.tunet = UNet(num_classes=unit, in_channels=feature_num[0] * unit, depth=3, start_filts=feature_num[1], dim=dim)

        self.branch_back_layers = nn.ModuleList(
            [
                ScaleBackConvLayer(unit, unit)
                for _ in range(branch_num)
            ]
        )

        self.tail = nn.ModuleList(

            [
                Conv3x3Block(in_channels=unit, out_channels=unit * feature_num[3], dim=dim, normalize=True),
                Conv3x3BlockTail(in_channels=unit * feature_num[3], out_channels=classes_num, dim=dim, normalize=True),
                # Conv3x3BlockTail(in_channels=unit, out_channels=classes_num, dim=dim, normalize=True)
            ]
        )

        self.channel_attn = nn.Parameter(
            torch.tensor([
                             1 / branch_num
                         ] * branch_num)
        )

    def forward(self, x):
        head_x = self.head(x)
        x_expand = self.branch_spilt(head_x)

        x_branch = []
        # split by channel
        for i in range(self.branch_num):
            x_branch.append(x_expand[:, i * self.unit: (i + 1) * self.unit, ...])

        x_branch_out = []
        middle_features = []
        params = []
        for x_slices, fl, bl in zip(x_branch, self.branch_forward_layers, self.branch_back_layers):
            x_s, param = fl(x_slices, x)
            short_cut = x_s
            # for exl in self.extract_layers:
            #     x_s, _ = exl(x_s)
            x_s = self.tunet(x_s)
            # x_s += short_cut

            # mid_res = x_s
            x_s, back_x = bl(x_s, param)
            # x_s = x_s + x_slices
            x_branch_out.append(x_s)
            middle_features.append(short_cut)
            params.append(param)

        res = 0
        i = 0
        for out_i in x_branch_out:
            input_ = out_i
            for layer in self.tail:
                input_ = layer(input_)
            # input_ += out_i
            res = input_ * self.channel_attn[i] + res
            i += 1
        # res = torch.softmax(res, dim=1)

        return res, middle_features, params


if __name__ == '__main__':
    sl = ScaleConvLayer(4, 4)
    model = ScaleSegmentNet()

    data = torch.randint(10, (2, 3, 512, 512), dtype=torch.float32)

    # data, param = sl(data)
    #
    # print(data.shape, param)

    res = model(data)

    for r in res:
        if isinstance(r, list):
            for rr in r:
                print(rr.shape)
        else:
            print(r.shape)
