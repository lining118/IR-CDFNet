import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

config = Config()
class PAM(nn.Module):
    def __init__(self, in_dim):
        super(PAM, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_dim*2,
            out_channels=2,
            kernel_size=3,
            padding=1
        )
        self.v_rgb = nn.Parameter(torch.randn((1, in_dim, 1, 1)), requires_grad=True)
        self.v_freq = nn.Parameter(torch.randn((1, in_dim, 1, 1)), requires_grad=True)
        #self.v_ori = nn.Parameter(torch.randn((1, in_dim, 1, 1)), requires_grad=True)

    def forward(self, rgb, freq):
        attmap = self.conv(torch.cat((rgb, freq), 1))
        attmap = torch.sigmoid(attmap)
        rgb = attmap[:, 0:1, :, :] * rgb * self.v_rgb
        freq = attmap[:, 1:, :, :] * freq * self.v_freq
        out = rgb + freq
        return out


class DAC(nn.Module):
    def __init__(self, in_dim):
        super(DAC, self).__init__()
        # 对关联性特征进行卷积操作
        self.conv1 = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        # 融合后的卷积操作
        self.fusion_conv = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        # 归一化层
        self.norm = nn.Softmax(dim=1)  # 在通道维度上进行归一化

    def forward(self, rgb, freq):
        # 计算RGB和频率特征的关联性
        relative_fact = rgb - freq
        relative_fact = self.conv1(relative_fact)
        # 对关联性特征进行归一化
        relative_fact = self.norm(relative_fact)
        # 使用归一化后的关联性特征作为卷积核分别对RGB和频率特征进行卷积
        rgb_out = rgb * relative_fact
        freq_out = freq * relative_fact
        # 将二者结果相加并进行融合卷积
        fused = self.fusion_conv(rgb_out + freq_out)
        return fused

class FRCv2(nn.Module):
    def __init__(self, in_dim):
        super(FRCv2, self).__init__()
        # 对关联性特征进行卷积操作
        self.conv1 = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        # 融合后的卷积操作
        self.fusion_conv = nn.Conv2d(2 * in_dim, 2 * in_dim, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(2*in_dim, in_dim,1)
        # 归一化层
        self.norm = nn.Softmax(dim=1)  # 在通道维度上进行归一化

    def forward(self, rgb, freq):
        # 计算RGB和频率特征的关联性
        relative_fact = rgb - freq
        relative_fact = self.conv1(relative_fact)
        # 对关联性特征进行归一化
        relative_fact = self.norm(relative_fact)
        # 使用归一化后的关联性特征作为卷积核分别对RGB和频率特征进行卷积
        rgb_out = rgb * relative_fact
        #freq_out = freq * relative_fact
        # 将二者结果相加并进行融合卷积
        fused =torch.cat((rgb_out, freq), 1)
        fused = self.fusion_conv(fused)
        fused = self.conv_out(fused)

        return fused

class MSF(nn.Module):
    # Partial Decoder Component (Search Module)
    def __init__(self):
        super(MSF, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(config.channels_list[2], config.channels_list[2], 3, padding=1)
        self.conv_up1_reshape = nn.Conv2d(config.channels_list[2], config.channels_list[1], 1)
        self.conv_upsample2 = BasicConv2d(config.channels_list[3], config.channels_list[3], 3, padding=1)
        self.conv_up2_reshape = nn.Conv2d(config.channels_list[3], config.channels_list[1], 1)

        self.conv_upsample3 = BasicConv2d(config.channels_list[2], config.channels_list[2], 3, padding=1)
        self.conv_up3_reshape = nn.Conv2d(config.channels_list[2], config.channels_list[1], 1)
        self.conv_upsample4 = BasicConv2d(config.channels_list[4], config.channels_list[4], 3, padding=1)
        self.conv_up4_reshape = nn.Conv2d(config.channels_list[4], config.channels_list[1], 1)

        #self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*config.channels_list[1], 2*config.channels_list[1], 3, padding=1)
        self.conv_concat3 = BasicConv2d(4*config.channels_list[1], 4*config.channels_list[1], 3, padding=1)
        self.conv4 = BasicConv2d(4*config.channels_list[1], 4*config.channels_list[1], 3, padding=1)
        self.conv5 = nn.Conv2d(4*config.channels_list[1], config.channels_list[1], 1)

    def forward(self, x1, x2, x3, x4):
        # print x1.shape, x2.shape, x3.shape, x4.shape
        x1_1 = x1
        x2_1 = self.conv_up1_reshape(self.conv_upsample1(self.upsample(x2))) * x1
        x3_1 = self.conv_up2_reshape(self.conv_upsample2(self.upsample(self.upsample(x3)))) * self.conv_up3_reshape(self.conv_upsample3(self.upsample(x2))) * x1

        x2_2 = torch.cat((x2_1, x1_1), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_up4_reshape(self.conv_upsample4(self.upsample(self.upsample(self.upsample(x4))))), x2_2), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)
        return x


class PDC_SE(nn.Module):
    # Partial Decoder Component (Search Edge)
    def __init__(self, channel):
        super(PDC_SE, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample2_1 = BasicConv2d(channel[2], channel[1], 3, padding=1)
        self.conv_upsample3_2 = BasicConv2d(channel[3], channel[2], 3, padding=1)
        self.conv_upsample2_3 = BasicConv2d(channel[2], channel[3], 3, padding=1)
        self.conv_upsample3_1 = BasicConv2d(channel[3], channel[1], 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel[2], channel[2], 3, padding=1)
        self.conv_upsample5 = BasicConv2d(channel[1] + channel[2], channel[1] + channel[2], 3, padding=1)

        self.conv_concat2 = BasicConv2d(channel[1] + channel[2], channel[1] + channel[2], 3, padding=1)
        self.conv_concat3 = BasicConv2d(2 * channel[1] + channel[2], 2 * channel[1] + channel[2], 3, padding=1)
        self.conv4 = BasicConv2d(2 * channel[1] + channel[2], 2 * channel[1] + channel[2], 3, padding=1)
        # self.conv5 = nn.Conv2d(2 * channel[1] + channel[2], 1, 1)

    def forward(self, x1, x2, x3):
        # print x1.shape, x2.shape, x3.shape, x4.shape
        # x1_1 = x1
        # print('e1.shape = {} \n e2.shape = {} \n e3.shape = {}'.format(x1.shape, x2.shape, x3.shape))
        x1_2 = self.conv_upsample2_1(self.upsample(x2)) * x1
        x2_3 = self.conv_upsample3_2(self.upsample(x3)) * x2
        x1_3 = self.conv_upsample3_1(self.upsample(self.upsample(x3)) * self.conv_upsample2_3(self.upsample(x2))) * x1

        # x2_3[1, 1152, 16, 16]  x1_2[1, 576, 32, 32]
        x1_2_3 = torch.cat((x1_2, self.conv_upsample4(self.upsample(x2_3))), 1)  # x1_2_3[1, 1728, 32, 32]
        x1_2_3 = self.conv_concat2(x1_2_3)

        # x1_3[1, 576, 32, 32])

        x_f = torch.cat((x1_3, x1_2_3), 1)
        x_f = self.conv_concat3(x_f)

        # x_f[1, 2 * channel[1] + channel[2], 32, 32]
        x = self.conv4(x_f)
        # x = self.conv5(x)

        return x

class NCD(nn.Module):
    def __init__(self, channel):
        super(NCD, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(x2_1)) * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
