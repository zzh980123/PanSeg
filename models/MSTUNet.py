from daformer import *
from MSFormer import *
import copy
# from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from decoder_unet import UnetDecoder
#################################################################


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
        decoder_dim_unet = [128, 64, 32, 16]
        
        # ----
        self.encoders_msformer = encoder(in_chans=in_channel)
        encoder_dim = [16, 32, 64, 128]
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


def run_check_net():
    batch_size = 2
    image_size = 800
    
    # ---
    batch = {
        'image': torch.from_numpy(np.random.uniform(-1, 1, (batch_size, 3, image_size, image_size))).float(),
        'mask': torch.from_numpy(np.random.choice(2, (batch_size, 1, image_size, image_size))).float(),
        'organ': torch.from_numpy(np.random.choice(5, (batch_size, 1))).long(),
    }
    batch = {k: v.cuda() for k, v in batch.items()}
    
    net = MSTUNet().cuda()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            output = net(batch)
    
    print('batch')
    for k, v in batch.items():
        print('%32s :' % k, v.shape)
    
    print('output')
    for k, v in output.items():
        if 'loss' not in k:
            print('%32s :' % k, v.shape)
    for k, v in output.items():
        if 'loss' in k:
            print('%32s :' % k, v.item())


# main #################################################################
if __name__ == '__main__':
    # run_check_net()
    input = torch.randn(16, 1, 512, 512)
    model = MSTUNet(in_channel=1, out_channel=1)
    output = model(input)
    print(output.shape)