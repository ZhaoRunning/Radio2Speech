import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from .TSB import TSB


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)

class down_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(down_conv, self).__init__()
        self.downconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True))

    def forward(self, downconv_input):
        return self.downconv(downconv_input)

# pixel shuffle
class up_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(True),
            nn.PixelShuffle(2))

        self.upconv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))

    def forward(self, upconv_input1, upconv_input2):
        upconv_input1 = self.up(upconv_input1)
        output = torch.cat([upconv_input1, upconv_input2], dim=1)
        output = self.upconv(output)
        return output


class out_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(out_conv, self).__init__()
        self.inoutconv = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))

    def forward(self, inout_input):
        return self.inoutconv(inout_input)

class in_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(in_conv, self).__init__()
        self.inconv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=(9,1), padding=(4,0)),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True),
                                nn.Conv2d(out_channels, out_channels, kernel_size=(1,9), padding=(0,4)),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
    def forward(self, input_amp):
        return self.inconv(input_amp)
    

def rescale_module(module):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv2d, nn.ConvTranspose2d)):
            # nn.init.xavier_uniform_(sub.weight)
            sub.weight.data.normal_(0.0, 0.001)
        elif isinstance(sub, nn.BatchNorm2d):
            sub.weight.data.normal_(1.0, 0.02)
            # sub.weight.data.fill_(1.)
            sub.bias.data.fill_(0)

class unet_TSB_encoder(nn.Module):
    def __init__(self, ngf=128, input_nc=1, rescale=True):
        super(unet_TSB_encoder, self).__init__()
        self.ngf = ngf
        self.rescale = rescale
        self.time_downsample_ratio = 2 ** 3 # This number equals 2^{#encoder_blcoks}

        #initialize layers
        self.inlayer = in_conv(input_nc, self.ngf)
        self.tsb_down1 = TSB(input_dim=80, in_channel=self.ngf, kernel_size=5, middle_kernel_size=25)
        self.downlayer1 = down_conv(self.ngf, self.ngf*2)
        self.tsb_down2 = TSB(input_dim=40, in_channel=self.ngf*2, kernel_size=5, middle_kernel_size=25)
        self.downlayer2 = down_conv(self.ngf*2, self.ngf*4)
        self.tsb_down3 = TSB(input_dim=20, in_channel=self.ngf*4, kernel_size=3, middle_kernel_size=15)
        self.downlayer3 = down_conv(self.ngf*4, self.ngf*8)
        # self.downlayer4 = down_conv(ngf*8, ngf*16)

        # weight initialization
        if self.rescale:
            rescale_module(self)

    def forward(self, radio_input):
        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = radio_input.size(-2)
        if not origin_len % self.time_downsample_ratio == 0:
            pad_len = int(np.ceil(origin_len / self.time_downsample_ratio)) \
                      * self.time_downsample_ratio - origin_len
            radio_input = F.pad(radio_input, pad=(0, 0, 0, pad_len))  #(bs, c, padded_time_steps, mel freq_bins)

        radio_feature = self.inlayer(radio_input)
        radio_feature_tsb = self.tsb_down1(radio_feature)
        radio_downfeature1 = self.downlayer1(radio_feature_tsb)
        radio_downfeature1_tsb = self.tsb_down2(radio_downfeature1)
        radio_downfeature2 = self.downlayer2(radio_downfeature1_tsb)
        radio_downfeature2_tsb = self.tsb_down3(radio_downfeature2)
        radio_downfeature3 = self.downlayer3(radio_downfeature2_tsb)
        # radio_downfeature4 = self.downlayer4(radio_downfeature3)
        features = [radio_downfeature2, radio_downfeature1, radio_feature]

        return radio_downfeature3, features, origin_len, radio_input

class unet_TSB_decoder(nn.Module):
    def __init__(self, hidden_size, ngf=128, output_nc=1, rescale=True):
        super(unet_TSB_decoder, self).__init__()
        self.rescale = rescale
        self.time_downsample_ratio = 2 ** 3

        head_channels = ngf*8 #512
        self.conv_more = Conv2dReLU(
            hidden_size,
            head_channels,
            kernel_size=1, #3
            # padding=1,
            use_batchnorm=True,
        )

        # self.uplayer1 = up_conv(ngf*16, ngf*8)
        self.uplayer2 = up_conv(ngf*8, ngf*4)
        self.uplayer3 = up_conv(ngf*4, ngf*2)
        self.uplayer4 = up_conv(ngf*2, ngf)
        self.outlayer = out_conv(ngf, output_nc)

        # weight initialization
        if self.rescale:
            rescale_module(self)

    def forward(self, hidden_states, encoder_features, origin_len, pad_input):
        # original input size to reconstruct feature for decoder
        # h = int(pad_input.size(-2) / 16)
        # w = int(pad_input.size(-1) / 16)

        h = int(pad_input.size(-2) / 8)
        w = int(pad_input.size(-1) / 8)

        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        down_feature = self.conv_more(x)

        # radio_upfeature1 = self.uplayer1(down_feature, encoder_features[0])
        radio_upfeature2 = self.uplayer2(down_feature, encoder_features[0])
        radio_upfeature3 = self.uplayer3(radio_upfeature2, encoder_features[1])
        radio_upfeature4 = self.uplayer4(radio_upfeature3, encoder_features[2])
        audio_output = self.outlayer(radio_upfeature4)

        if not origin_len % self.time_downsample_ratio == 0:
            audio_output = audio_output[..., : origin_len, :] # (bs, channels, T, F)

        assert audio_output.size(-2) == origin_len

        return audio_output