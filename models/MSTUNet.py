from daformer import *
from MSFormer import *
import copy
# from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from decoder_unet import UnetDecoder
#################################################################


class SingleUpSample(nn.Module):
    def __init__(self, scale_factor=2, mode='bilinear'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class MixUpSample(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.mixing = nn.Parameter(torch.tensor(0.5))
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.mixing * F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False) \
            + (1 - self.mixing) * F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        return x


class MSTUNet(nn.Module):
    def __init__(self,
                 in_channel=3,
                 out_channel=3,
                 encoder=MSFormer,
                 encoder_ckpt=None,
                 ):
        super(MSTUNet, self).__init__()
        decoder_dim_unet = [512, 256, 128, 64]
        
        # ----
        embed_dims = [32, 64, 128, 256]
        encoder_dim = [64, 128, 256, 512]
        self.encoders_msformer = encoder(in_chans=in_channel, embed_dims=embed_dims)

        # [64, 128, 320, 512]
        
        self.decoders_unet = torch.nn.ModuleList()
        self.decoders_unet = \
            UnetDecoder(
                encoder_channels=encoder_dim,
                decoder_channels=decoder_dim_unet,
                n_blocks=4,
                use_batchnorm=True,
                center=False,
                attention_type=None,
            )

        self.logit_unet = nn.Sequential(
            nn.Conv2d(decoder_dim_unet[-1], out_channel, kernel_size=1),
        )
        self.interpolate = MixUpSample(scale_factor=4)
        # self.interpolate = SingleUpSample(scale_factor=4, mode='nearest')

    def forward(self, x):
        B, C, H, W = x.shape

        encoder = self.encoders_msformer(x)

        feature = encoder[::-1]
        head = feature[0]
        skip = feature[1:]
        d = head
        
        decoder = []
        for i, decoder_block in enumerate(self.decoders_unet.blocks):
            s = skip[i]
            d = decoder_block(d, s)
            decoder.append(d)
        last = d

        logit = self.logit_unet(last)

        # upsample_logit = F.interpolate(logit, size=None, scale_factor=4, mode='bilinear', align_corners=False)
        upsample_logit = self.interpolate(logit)
        return upsample_logit


if __name__ == '__main__':
    input = torch.randn(16, 1, 512, 512)
    model = MSTUNet(in_channel=1, out_channel=1)
    output = model(input)
    print(output.shape)