import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from einops import rearrange
import flow
import common
import layers

class TSFNet(nn.Module):
    def __init__(self, device=None, in_channels=1, hidden_dim=32, out_channels=4, max_scaling=4):
        super(TSFNet, self).__init__()

        self.max_scaling = max_scaling
        self.min_scaling = 1 / max_scaling
        assert self.max_scaling > 0
        self.M = math.log2(self.max_scaling)

        self.conv_span = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv_reduce1 = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)

        self.normal1 = nn.BatchNorm2d(hidden_dim)

        self.conv_reduce2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, padding=0)
        self.reg_inv = common.Involution(channels=hidden_dim, kernel_size=7, stride=1)

        self.reg = nn.Conv2d(in_channels=out_channels, out_channels=4, kernel_size=1, padding=0)
        self.max_pooling = nn.AdaptiveMaxPool2d(1, return_indices=True)

        self.flatten_pt = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1, padding=0)

        self.later_conv1 = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.later_conv2 = nn.Conv2d(4, 1, kernel_size=1, padding=0)

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

    def forward(self, x):
        ori_shape = x.shape
        ori_feature = x

        x = self.conv_span(x)  # (b, 32, 512, 512)
        feat = self.conv_reduce1(x)  # h (b, 4, 512, 512)
        feat = F.leaky_relu(feat)

        x = self.normal1(x)

        params = self.conv_reduce2(self.reg_inv(x))  # params (b, 4, 512, 512)

        b, c, h, w = params.shape
        _, max_index = self.max_pooling(params[:, 3:, ...])  # 最后一个通道的最大值序号
        max_index = torch.repeat_interleave(max_index, c, dim=1).view((b, c, 1))
        flatten_feature = rearrange(params, 'b c h w -> b c (h w) 1')
        idx1 = torch.arange(b).view(-1, 1, 1)
        idx2 = torch.arange(c).view(1, -1, 1)
        params = flatten_feature[idx1, idx2, max_index]  # 取每个Batch和Channel下max_index所在值  (b, 4, 1)

        xy, s, _ = params[:, 0:2, ...], params[:, 2:3, ...], params[:, 3:, ...]

        xy_normal = torch.sigmoid(xy) - 0.5
        s_normal = torch.exp2(self.M * (2 * torch.sigmoid(s) - 1))

        for i in range(xy_normal.shape[1]):
            xy_normal[:, i, ...] *= ori_shape[i + 2]  # 位移变换场需乘上原图的h w

        s_flow, t_flow = self.get_st_flows(ori_shape, x.device)
        s_flow = flow.trans_s_flow(s_flow, s_normal)
        t_flow = flow.trans_t_flow(t_flow, xy_normal)

        flatten_ps = self.flatten_pt(feat)

        trans_fps = self.trans(self.trans(flatten_ps, t_flow), s_flow)
        trans_fps = self.later_conv2(self.later_conv1(trans_fps))
        trans_fps = self.normal2(trans_fps)

        return trans_fps, params




if __name__ == '__main__':
    input = torch.randn((4, 1, 512, 512))
    model = TSFNet()
    _, _ = model(input)
    




