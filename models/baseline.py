import torch
import torch.nn as nn
from einops import rearrange
from config import Config
from models.backbones.build_backbone import build_backbone
from torchvision.models import ResNet50_Weights
import torchvision.transforms as transforms


from models.modules.lateral_blocks import two_ConvBnRule
from models.modules.enhanced_blocks import FSF
from models.modules.attention_blocks import HOR
from models.modules.head_blocks import ReverseStage, CentralReviseHeaBlk
from utils.utils import save_tensor_img
from models.modules.mix_blocks import PAM, MSF, DAC, FRCv2

from models.Res2Net_v1b import res2net50_v1b

from utils.utils import Seg, norm
import torch_dct as DCT
import torch.nn.functional as F

# 生成随机张量
random_tensor1 = torch.rand(1, 64, 1, 1)

# 将范围从 [0, 1) 转换为 [-1, 1)
scaled_tensor1 = 2 * random_tensor1 - 1

random_tensor2 = torch.rand(1, 64, 1, 1)

# 将范围从 [0, 1) 转换为 [-1, 1)
scaled_tensor2 = 2 * random_tensor2 - 1

random_tensor3 = torch.rand(1, 64, 1, 1)

# 将范围从 [0, 1) 转换为 [-1, 1)
scaled_tensor3 = 2 * random_tensor3 - 1

class FDNet(nn.Module):
    def __init__(self, backbone_pretrained=True):
        super(FDNet, self).__init__()
        self.config = Config()
        self.epoch = 1
        self.backbone = build_backbone(self.config.backbone, self.config.is_pretrained_backbone)
        self.decoder = Decoder(self.config.channels_list)

        self.resnet = res2net50_v1b(pretrained=True)
        self.res_con1 = self.resnet.conv1
        self.res_bn1 = self.resnet.bn1
        self.res_relu = self.resnet.relu
        self.res_mxpool = self.resnet.maxpool
        self.res_layer1 = self.resnet.layer1
        self.res_layer2 = self.resnet.layer2
        self.res_layer3 = self.resnet.layer3
        self.res_layer4 = self.resnet.layer4
    def forward_preprocess(self, x, x_dct):
        if self.config.backbone in ['vgg16', 'vgg16bn', 'resnet50']:
            x1 = self.backbone.conv1(x)
            x2 = self.backbone.conv2(x1)
            x3 = self.backbone.conv3(x2)
            x4 = self.backbone.conv4(x3)
        elif self.config.backbone in ['MambaVision_b_1k', 'MambaVision_l_1k', 'MambaVision_s_1k']:
            _, outs = self.backbone(x)
            x1, x2, x3, x4 = outs

            if self.config.mul_scl_ipt == 'cat':
                B, C, H, W = x.shape
                _, outs_ = self.backbone(
                    F.interpolate(x, size=(H // 2, W // 2), mode='bilinear', align_corners=True))
                x1_, x2_, x3_, x4_ = outs_
                x1 = torch.cat([x1, F.interpolate(x1_, size=x1.shape[2:], mode='bilinear', align_corners=True)], dim=1)
                x2 = torch.cat([x2, F.interpolate(x2_, size=x2.shape[2:], mode='bilinear', align_corners=True)], dim=1)
                x3 = torch.cat([x3, F.interpolate(x3_, size=x3.shape[2:], mode='bilinear', align_corners=True)], dim=1)
                x4 = torch.cat([x4, F.interpolate(x4_, size=x4.shape[2:], mode='bilinear', align_corners=True)], dim=1)
            elif self.config.mul_scl_ipt == 'add':
                B, C, H, W = x.shape
                _, outs_ = self.backbone(
                    F.interpolate(x, size=(H // 2, W // 2), mode='bilinear', align_corners=True))
                x1_, x2_, x3_, x4_ = outs_
                x1 = x1 + F.interpolate(x1_, size=x1.shape[2:], mode='bilinear', align_corners=True)
                x2 = x2 + F.interpolate(x2_, size=x2.shape[2:], mode='bilinear', align_corners=True)
                x3 = x3 + F.interpolate(x3_, size=x3.shape[2:], mode='bilinear', align_corners=True)
                x4 = x4 + F.interpolate(x4_, size=x4.shape[2:], mode='bilinear', align_corners=True)
            if self.config.mul_lev_ipt:
                x4_ = x4
                x4 = torch.cat(
                    (
                        *[
                             F.interpolate(x1, size=x4.shape[2:], mode='bilinear', align_corners=True),






                             F.interpolate(x2, size=x4.shape[2:], mode='bilinear', align_corners=True),
                             F.interpolate(x3, size=x4.shape[2:], mode='bilinear', align_corners=True),
                         ][-self.config.mul_scl_sc_num:],
                        x4
                    ),
                    dim=1
                )
                x1 = torch.cat(
                    (
                        F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=True),
                        x1
                    ),
                    dim=1
                )
                x2 = torch.cat(
                    (
                        F.interpolate(x3, size=x2.shape[2:], mode='bilinear', align_corners=True),
                        x2
                    ),
                    dim=1
                )
                x3 = torch.cat(
                    (
                        F.interpolate(x4_, size=x3.shape[2:], mode='bilinear', align_corners=True),
                        x3
                    ),
                    dim=1
                )
        else:
            x1, x2, x3, x4 = self.backbone(x)


        return x, x_dct, x1, x2, x3, x4

    def forward(self, x, x_dct):
        import os
        import torch

        def save_tensors_if_not_exist(x, x_dct, save_path="input_images.pth"):
            """
            检查是否存在保存的 .pth 文件，如果不存在，则保存 x 和 x_dct。

            参数:
                x (torch.Tensor): 空间域输入图像 (B, C, H, W)
                x_dct (torch.Tensor): DCT 变换后的输入图像 (B, C, H, W)
                save_path (str): 保存文件路径，默认为 "input_images.pth"
            """

            # 如果文件不存在，则进行保存
            if not os.path.exists(save_path):
                print(f"保存输入数据到 {save_path}...")
                torch.save({'x': x.detach().cpu(), 'x_yxbcr': x_dct.detach().cpu()}, save_path)
            else:
                print(f"文件 {save_path} 已存在，跳过保存。")

        save_tensors_if_not_exist(x, x_dct)  # 调用检查并保存的函数

        ########## Encoder ##########
        x, x_dct, x1, x2, x3, x4 = self.forward_preprocess(x, x_dct)
        ########## Decoder ##########
        features = [x, x_dct, x1, x2, x3, x4]
        output_1, output_2, output_3, output_4 = self.decoder(
            features)
        return output_1, output_2, output_3, output_4


class Decoder(nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()
        self.config = Config()
        LateralBlock = eval(self.config.lat_blk)
        HeadBlock = eval(self.config.hea_blk)
        # AttentionBlock = eval(self.config.att_blk)
        MixBlock2 = MSF
        EnhancedBlock = eval(self.config.enh_blk)
        self.lateral_block1 = LateralBlock(channels[1], channels[1])
        self.lateral_block2 = LateralBlock(channels[2], channels[2])
        self.lateral_block3 = LateralBlock(channels[3], channels[3])
        self.lateral_block4 = LateralBlock(channels[4], channels[4])
        self.lateral_block5_1 = LateralBlock(channels[1], channels[1])
        self.lateral_block5_2 = LateralBlock(channels[2], channels[2])
        self.lateral_block5_3 = LateralBlock(channels[3], channels[3])
        self.lateral_block5_4 = LateralBlock(channels[4], channels[4])

        MixBlock1 = eval(self.config.mix_blk)
        self.mix_block1 = MixBlock1(channels[1])
        self.mix_block2 = MixBlock1(channels[2])
        self.mix_block3 = MixBlock1(channels[3])
        self.mix_block4 = MixBlock1(channels[4])

        self.enchanced_block1 = EnhancedBlock(channels[1])
        self.enchanced_block2 = EnhancedBlock(channels[2])
        self.enchanced_block3 = EnhancedBlock(channels[3])
        self.enchanced_block4 = EnhancedBlock(channels[4])

        self.mix_pdc = MixBlock2()

        self.head_block1 = HeadBlock(channels[1])
        self.head_block2 = HeadBlock(channels[1])
        self.head_block3 = HeadBlock(channels[1])

        #self.lateral_block_f1 = LateralBlock(channels[1], channels[1])
        #self.lateral_block_f2 = LateralBlock(channels[1], channels[1])

        # 1*1 conv
        self.con1_2 = nn.Conv2d(in_channels=192, out_channels=channels[1], kernel_size=1)
        self.con1_3 = nn.Conv2d(in_channels=192, out_channels=channels[2], kernel_size=1)
        self.con1_4 = nn.Conv2d(in_channels=192, out_channels=channels[3], kernel_size=1)
        self.con1_5 = nn.Conv2d(in_channels=192, out_channels=channels[4], kernel_size=1)

        self.seg = Seg().to(self.config.device)

        self.conv_out1 = nn.Conv2d(192,1,1)
        self.conv_out2 = nn.Conv2d(192,1,1)
        self.conv_out3 = nn.Conv2d(192,1,1)
        self.conv_out4 = nn.Conv2d(192,1,1)

    def forward(self, features):
        # fds = []
        x, x_dct, x1, x2, x3, x4 = features
        #print('x1.shape = {}'.format(x1.shape))
        #print('x2.shape = {}'.format(x2.shape))
        #print('x3.shape = {}'.format(x3.shape))
        #print('x4.shape = {}'.format(x4.shape))

        # 横向卷积
        x1 = self.lateral_block1(x1)
        x2 = self.lateral_block2(x2)
        x3 = self.lateral_block3(x3)
        x4 = self.lateral_block4(x4)

        # FSF
        feat_DCT_1 = self.enchanced_block1(x_dct, x1)
        feat_DCT_2 = self.enchanced_block2(x_dct, x2)
        feat_DCT_3 = self.enchanced_block3(x_dct, x3)
        feat_DCT_4 = self.enchanced_block4(x_dct, x4)

        # 将FSF的输出特征图变换到x1，x2，x3，x4 的尺寸，con变换通道。interpolate变换分辨率
        feat_DCT_1 = self.con1_2(feat_DCT_1)
        feat_DCT_2 = self.con1_3(feat_DCT_2)
        feat_DCT_3 = self.con1_4(feat_DCT_3)
        feat_DCT_4 = self.con1_5(feat_DCT_4)
        feat_DCT_1 = torch.nn.functional.interpolate(feat_DCT_1, size=x1.size()[2:], mode='bilinear',align_corners=True)
        feat_DCT_2 = torch.nn.functional.interpolate(feat_DCT_2, size=x2.size()[2:], mode='bilinear',align_corners=True)
        feat_DCT_3 = torch.nn.functional.interpolate(feat_DCT_3, size=x3.size()[2:], mode='bilinear', align_corners=True)
        feat_DCT_4 = torch.nn.functional.interpolate(feat_DCT_4, size=x4.size()[2:], mode='bilinear',align_corners=True)

        # DAC
        x1 = self.mix_block1(x1, feat_DCT_1)
        x2 = self.mix_block2(x2, feat_DCT_2)
        x3 = self.mix_block3(x3, feat_DCT_3)
        x4 = self.mix_block4(x4, feat_DCT_4)

        x1 = self.lateral_block5_1(x1)
        x2 = self.lateral_block5_2(x2)
        x3 = self.lateral_block5_3(x3)
        x4 = self.lateral_block5_4(x4)
        coarse_m1 = self.mix_pdc(x1, x2, x3, x4)

        # reverse 1
        coarse_m2, guide1 = self.head_block1(coarse_m1, x)

        #coarse_m2 = self.lateral_block_f1(coarse_m2)

        # reverse 2
        coarse_m3, guide2 = self.head_block2(coarse_m2, x)

        #coarse_m3 = self.lateral_block_f2(coarse_m3)

        # reverse 3
        coarse_m4, guide3 = self.head_block3(coarse_m3, x)

        # outputs
        #coarse_m1 = self.conv_out1(coarse_m1)
        #coarse_m2 = self.conv_out2(coarse_m2)
        #coarse_m3 = self.conv_out3(coarse_m3)
        coarse_m4 = self.conv_out4(coarse_m4)

        sizex = x.size()[2:]
        output_1 = torch.nn.functional.interpolate(guide1, size=sizex, mode='bilinear', align_corners=True)
        output_2 = torch.nn.functional.interpolate(guide2, size=sizex, mode='bilinear', align_corners=True)
        output_3 = torch.nn.functional.interpolate(guide3, size=sizex, mode='bilinear', align_corners=True)
        output_4 = torch.nn.functional.interpolate(coarse_m4, size=sizex, mode='bilinear', align_corners=True)
        return output_1, output_2, output_3, output_4



def check_tensor(tensor, name):
    if torch.isnan(tensor).any():
        print(f"{name} 中包含 NaN 值")
    if torch.isinf(tensor).any():
        print(f"{name} 中包含 Inf 值")
    print(
        f"{name} 的统计数据: mean={tensor.mean().item()}, std={tensor.std().item()}, min={tensor.min().item()}, max={tensor.max().item()}")


if __name__ == "__main__":
    x = torch.randn(4, 3, 512, 512).to(0)
    y = torch.randn(4, 3, 512, 512).to(0)
    # detail = Detail_Branch()
    # feat = detail(x)
    # print('detail', feat.size())

    net = FDNet()
    logits = net(x, y)
    #print('logits = {}'.format(logits))
