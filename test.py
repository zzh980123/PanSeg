import torch




if __name__ == '__main__':
    x = torch.rand((32, 4, 16, 16))
    b, c, h, w = x.shape
    flatten_x = x.view((b, c, h * w))
    top_value, top_index = torch.topk(flatten_x[:, 3:, ...], 3)
    idx1 = torch.arange(b).view(-1, 1, 1)
    idx2 = torch.arange(c).view(1, -1, 1)
    params = flatten_x[idx1, idx2, top_index]
    mean_params = torch.mean(params, -1).view(b, c, 1)
    print(top_value)
