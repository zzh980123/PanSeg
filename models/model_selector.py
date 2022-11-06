import monai
from monai.networks.nets import *
from monai.networks.blocks import Warp
from daformer_coat_net import *


def model_factory(model_name: str, device, args, in_channels=1, spatial_dims=2, pretrained_model_paths=[]):
    if model_name == 'unet':
        model = monai.networks.nets.UNet(  # type: ignore
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=args.num_class,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(device)
    if model_name == 'vnet':
        model = monai.networks.nets.VNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=args.num_class,
            dropout_prob=0.0
        ).to(device)
    if model_name == 'swinunetr':
        model = SwinUNETR(
            img_size=(args.input_size, args.input_size),
            in_channels=in_channels,
            out_channels=args.num_class,
            feature_size=24,  # should be divisible by 12
            spatial_dims=spatial_dims
        ).to(device)
    if model_name == 'unetr':
        model = UNETR(
            in_channels=in_channels,
            out_channels=args.num_class,
            img_size=(args.input_size, args.input_size),
            feature_size=24,
            spatial_dims=spatial_dims
        ).to(device)
    if model_name == 'coat':
        model = DaFormaerCoATNet(
            in_channel=in_channels,
            out_channel=args.num_class,
        ).to(device)
    return model
