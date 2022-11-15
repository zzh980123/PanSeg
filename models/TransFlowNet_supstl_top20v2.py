import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from einops import rearrange
import flow
import common
from model_selector import *
import layers


class TransBack(nn.Module):
    def __init__(self, in_channels, out_channels, max_scaling=4):
        super(TransBack, self).__init__()
        self.max_scaling = max_scaling
        self.min_scaling = 1 / max_scaling
        self.M = math.log2(max_scaling)
        self.dyn_flow_shape = True
        self.later_conv = HugeConv2dBlock(in_channels, out_channels)

    def get_st_flows(self, shape_, device):
        if self.dyn_flow_shape:
            self.register_buffer('s_flow', nn.Parameter(flow.gen_flow_scale(shape_[2:]), requires_grad=False))
            self.register_buffer('t_flow', nn.Parameter(flow.gen_flow_transport(shape_[2:]), requires_grad=False))
            self.trans = flow.SpatialTransformer(shape_[2:], mode='nearest')

            self.s_flow = self.s_flow.to(device)
            self.t_flow = self.t_flow.to(device)
            self.dyn_flow_shape = False
        return self.s_flow, self.t_flow

    def forward(self, x, xy_normal, s_normal):
        shape_ = x.shape
        for i in range(xy_normal.shape[1]):
            xy_normal[:, i, ...] *= shape_[i + 2]

        s_flow, t_flow = self.get_st_flows(shape_, x.device)
        s_flow = flow.trans_s_flow(s_flow, 1 / s_normal)
        t_flow = flow.trans_t_flow(t_flow, -1 * xy_normal)
        trans_back = self.trans(self.trans(x, s_flow), t_flow)

        output = self.later_conv(trans_back)

        return output


class TransForward(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=16, out_channels=2, max_scaling=4):
        super(TransForward, self).__init__()

        self.h = [2]
        self.max_scaling = max_scaling
        self.min_scaling = 1 / max_scaling
        assert self.max_scaling > 0
        self.M = math.log2(self.max_scaling)

        self.conv_span = nn.Conv2d(in_channels, hidden_dim // 2, kernel_size=3, padding=1)
        self.downSample1 = common.Down(hidden_dim // 2, hidden_dim)
        self.downSample2 = common.Down(hidden_dim, hidden_dim * 2)
        self.downSample3 = common.Down(hidden_dim * 2, hidden_dim * 4)
        self.downSample4 = common.Down(hidden_dim * 4, hidden_dim * 8)
        self.downSample5 = common.Down(hidden_dim * 8, hidden_dim * 16)
        self.upSample0 = common.Up(hidden_dim * 16, hidden_dim * 8, False)
        self.upSample1 = common.Up(hidden_dim * 8, hidden_dim * 4, False)
        self.upSample2 = common.Up(hidden_dim * 4, hidden_dim * 2, False)
        self.upSample3 = common.Up(hidden_dim * 2, hidden_dim, False)
        self.upSample4 = common.Up(hidden_dim, out_channels, False)

        self.max_pooling = nn.AdaptiveMaxPool2d(1, return_indices=True)

        self.flatten_pt = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1, padding=0)

        self.normal2 = nn.BatchNorm2d(1)

        self.dyn_flow_shape = True

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

    def forward(self, x, label):
        ori_shape = x.shape
        ori_feature = x

        # x = self.conv_span(x)  # (b, 16, 512, 512)
        # # involution
        # x = self.normal1(x)
        # params = self.conv_reduce(self.reg_inv(x))  # params (b, 4, 512, 512)

        # unet-like down4
        x = self.conv_span(x)  # (b, 8, 512, 512)
        d1 = self.downSample1(x)  # (b, 16, 256, 256)
        d2 = self.downSample2(d1)  # (b, 32, 128, 128)
        d3 = self.downSample3(d2)  # (b, 64, 64, 64)
        d4 = self.downSample4(d3)  # (b, 128, 32, 32)
        d5 = self.downSample5(d4)

        u0 = self.upSample0(d5, d4)
        u1 = self.upSample1(u0, d3)  # (b, 64, 64, 64)
        u2 = self.upSample2(u1, d2)  # (b, 32, 128, 128)
        u3 = self.upSample3(u2, d1)  # (b, 16, 256, 256)
        features = self.upSample4(u3, x)  # (b, 2, 512, 512)          c:0 coarse seg        c:1 learned scale size
        coarse_seg = features[:, 0, ...].unsqueeze(1)

        xy_normal, s_normal = flow.get_normal_image(features, 20, self.M)

        s_flow, t_flow = self.get_st_flows(ori_shape, x.device)
        s_flow = flow.trans_s_flow(s_flow, s_normal)
        t_flow = flow.trans_t_flow(t_flow, xy_normal)

        trans_f = self.trans(self.trans(ori_feature, t_flow), s_flow)
        trans_f_label = self.trans(self.trans(label, t_flow), s_flow)
        return trans_f, trans_f_label, xy_normal, s_normal, coarse_seg


class TransFlowNet(nn.Module):
    def __init__(self, model_name: str, device, args, in_channels=1, out_channels=1, hidden_dim=16, spatial_dims=2):
        super(TransFlowNet, self).__init__()

        self.model = model_factory(model_name, device, args, in_channels)
        self.trans_forward = TransForward(in_channels, hidden_dim=hidden_dim, out_channels=2, max_scaling=4).to(device)
        self.trans_back = TransBack(in_channels=in_channels, out_channels=out_channels).to(device)

    def forward(self, x, label):
        trans_f, trans_f_label, xy_normal, s_normal, coarse_seg = self.trans_forward(x, label)
        hidden_feature = self.model(trans_f)
        output = self.trans_back(hidden_feature, xy_normal, s_normal)
        return trans_f, trans_f_label, hidden_feature, output, xy_normal, s_normal, coarse_seg

