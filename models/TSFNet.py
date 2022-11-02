import math

import monai.networks.nets
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import *
from einops import rearrange
import flow

class TSFNet(nn.Module):
    def __init__(self, device=None, in_channels=1, out_channels=4, max_scaling=4):
        super(TSFNet, self).__init__()
        self.head = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2
        )
        self.max_scaling = max_scaling
        self.min_scaling = 1 / max_scaling
        assert self.max_scaling > 0
        self.M = math.log2(self.max_scaling)
        
        self.max_pooling = nn.AdaptiveMaxPool2d(1, return_indices=True)

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
        feature = self.head(x)
        feature = torch.sigmoid(feature)
        
        b, c, h, w = feature.shape
        _, max_index = self.max_pooling(feature[:, 3:, ...])  # 最后一个通道的最大值序号
        max_index = torch.repeat_interleave(max_index, c, dim=1).view((b, c, 1))
        flatten_feature = rearrange(feature, 'b c h w -> b c (h w)')
        idx1 = torch.arange(b).view(-1, 1, 1)
        idx2 = torch.arange(c).view(1, -1, 1)
        params = flatten_feature[idx1, idx2, max_index]  # 取每个Batch和Channel下max_index所在值

        xy, s, _ = params[:, 0:2, ...], params[:, 2:3, ...], params[:, 3:, ...]

        xy_normal = torch.sigmoid(xy) - 0.5
        s_normal = torch.exp2(self.M * (2 * torch.sigmoid(s) - 1))

        for i in range(c):
            xy_normal[:, i, ...] *= ori_shape[i + 2]  # 位移变换场需乘上原图的h w

        s_flow, t_flow = self.get_st_flow(ori_shape, x.device)


        return x


        
        
        
        
        
        

if __name__ == '__main__':
    input = torch.randn((4, 1, 512, 512))
    model = TSFNet()
    output = model(input)
    




