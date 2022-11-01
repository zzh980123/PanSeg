import torch
import torch.nn as nn
import torch.nn.functional as F

class TSFNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, branch_nums=3, depth=[0, 0, 0, 0], dims=[32, 64, 128, 256], backbone=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.branch_nums = branch_nums



