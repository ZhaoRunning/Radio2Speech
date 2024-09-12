import torch.nn as nn
import torch

class FTB(nn.Module):
    def __init__(self, input_dim=80, in_channel=32, r_channel=5):
        super(FTB, self).__init__()
        self.in_channel = in_channel
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_channel, r_channel, kernel_size=(1, 1)),
        #     nn.BatchNorm2d(r_channel),
        #     nn.ReLU())
        # self.conv1d = nn.Sequential(
        #     nn.Conv1d(r_channel * input_dim, in_channel, kernel_size=9, padding=4),
        #     nn.BatchNorm1d(in_channel),
        #     nn.ReLU())
        self.freq_fc = nn.Linear(input_dim, input_dim, bias=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel, kernel_size=(1, 1)),
            nn.BatchNorm2d(in_channel),
            nn.ReLU())

    def forward(self, inputs):
        '''
        inputs should be [Batch, Ca, Dim, Time]
        '''
        # T-F attention

        # conv1_out = self.conv1(inputs)
        # B, C, D, T = conv1_out.size()
        # reshape1_out = torch.reshape(conv1_out, [B, C * D, T])
        # conv1d_out = self.conv1d(reshape1_out)
        # conv1d_out = torch.reshape(conv1d_out, [B, self.in_channel, 1, T])
        #
        # # now is also [B,C,D,T]
        # att_out = conv1d_out * inputs

        att_out = inputs
        # tranpose to [B,C,T,D]
        att_out = torch.transpose(att_out, 2, 3)
        freqfc_out = self.freq_fc(att_out)
        att_out = torch.transpose(freqfc_out, 2, 3)

        cat_out = torch.cat([att_out, inputs], 1)
        outputs = self.conv2(cat_out)

        return outputs

class TSB(nn.Module):
    # source code: https://github.com/huyanxin/phasen/blob/master/model/phasen.py
    def __init__(self, input_dim=80, in_channel=32, kernel_size=5, middle_kernel_size=25):
        super(TSB, self).__init__()

        # self.ftb1 = FTB(input_dim=input_dim, in_channel=in_channel)
        self.amp_conv1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=(kernel_size, kernel_size),
                      padding=(kernel_size//2, kernel_size//2)),
            nn.BatchNorm2d(in_channel),
            nn.ReLU())
        # capture long-range time-domain correlation
        self.amp_conv2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=(middle_kernel_size, 1), padding=(middle_kernel_size//2, 0)),
            nn.BatchNorm2d(in_channel),
            nn.ReLU())
        self.amp_conv3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=(kernel_size, kernel_size),
                      padding=(kernel_size//2, kernel_size//2)),
            nn.BatchNorm2d(in_channel),
            nn.ReLU())
        self.ftb2 = FTB(input_dim=input_dim, in_channel=in_channel)

    def forward(self, amp):
        '''
        amp should be [Batch, Ca, Dim, Time]
        amp should be [Batch, Cr, Dim, Time]
        '''

        amp_out1 = amp.transpose(-1, -2)
        # amp_out1 = self.ftb1(amp)
        amp_out2 = self.amp_conv1(amp_out1)
        amp_out3 = self.amp_conv2(amp_out2)
        amp_out4 = self.amp_conv3(amp_out3)
        amp_out5 = self.ftb2(amp_out4)
        amp_out5 = amp_out5.transpose(-1, -2)

        return amp_out5