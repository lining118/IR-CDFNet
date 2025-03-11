import torch
import torch.nn as nn
from config import Config
import torch.nn.functional as F
from models.modules.decoder_blocks import RF, BasicDecBlk
from models.modules.lateral_blocks import two_ConvBnRule
from utils.utils import path_to_image, save_tensor_img  # 导入工具函数
from PIL import Image
import numpy as np



config = Config()


class OutHeaBlk(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutHeaBlk, self).__init__()
        #output
    def forward(self, x):
        return x


class AttBeaBlk(nn.Module):
    def __init__(self, in_channels, out_channels): # in_channels = weight map channels + feature map channels
        super(AttBeaBlk, self).__init__()
        # 注意力模块，用于融合边缘特征和分割特征
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 1, kernel_size=1),  # 输出一个注意力权重
            nn.Sigmoid()  # 注意力权重归一化到 [0, 1] 之间
        )

    def forward(self, pred_e, sm):
        # 使用双线性插值将边缘特征与分割特征调整到相同尺寸
        # pred_e = F.interpolate(pred_e, size=sm.shape[2:], mode='bilinear', align_corners=False)

        # 融合边缘特征和分割特征
        combined_features = torch.cat((pred_e, sm), dim=1)
        attention_weights = self.attention(combined_features)

        # 对分割特征应用注意力权重
        pred_m = sm * attention_weights

        return pred_m



# Group-Reversal Attention (GRA) Block
class GRA(nn.Module):
    def __init__(self, channel, subchannel):
        super(GRA, self).__init__()
        self.group = channel//subchannel
        self.conv = nn.Sequential(
            nn.Conv2d(channel + self.group, channel, 3, padding=1), nn.ReLU(True),
        )
        self.score = nn.Conv2d(channel, 1, 3, padding=1)

    def forward(self, x, y):
        if self.group == 1:
            x_cat = torch.cat((x, y), 1)
        elif self.group == 2:
            xs = torch.chunk(x, 2, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y), 1)
        elif self.group == 4:
            xs = torch.chunk(x, 4, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y), 1)
        elif self.group == 8:
            xs = torch.chunk(x, 8, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y), 1)
        elif self.group == 16:
            xs = torch.chunk(x, 16, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y,
            xs[8], y, xs[9], y, xs[10], y, xs[11], y, xs[12], y, xs[13], y, xs[14], y, xs[15], y), 1)
        elif self.group == 32:
            xs = torch.chunk(x, 32, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y,
            xs[8], y, xs[9], y, xs[10], y, xs[11], y, xs[12], y, xs[13], y, xs[14], y, xs[15], y,
            xs[16], y, xs[17], y, xs[18], y, xs[19], y, xs[20], y, xs[21], y, xs[22], y, xs[23], y,
            xs[24], y, xs[25], y, xs[26], y, xs[27], y, xs[28], y, xs[29], y, xs[30], y, xs[31], y), 1)
        elif self.group == 64:
            xs = torch.chunk(x, 64, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y,
                               xs[8], y, xs[9], y, xs[10], y, xs[11], y, xs[12], y, xs[13], y, xs[14], y, xs[15], y,
                               xs[16], y, xs[17], y, xs[18], y, xs[19], y, xs[20], y, xs[21], y, xs[22], y, xs[23], y,
                               xs[24], y, xs[25], y, xs[26], y, xs[27], y, xs[28], y, xs[29], y, xs[30], y, xs[31], y,
                               xs[32], y, xs[33], y, xs[34], y, xs[35], y, xs[36], y, xs[37], y, xs[38], y, xs[39], y,
                               xs[40], y, xs[41], y, xs[42], y, xs[43], y, xs[44], y, xs[45], y, xs[46], y, xs[47], y,
                               xs[48], y, xs[49], y, xs[50], y, xs[51], y, xs[52], y, xs[53], y, xs[54], y, xs[55], y,
                               xs[56], y, xs[57], y, xs[58], y, xs[59], y, xs[60], y, xs[61], y, xs[62], y, xs[63], y,
                               ), 1)
        elif self.group == 49:
            xs = torch.chunk(x, 49, dim=1)
            x_cat = torch.cat([xs[i // 2] if i % 2 == 0 else y for i in range(49*2)], dim=1)
        elif self.group == 98:
            xs = torch.chunk(x, 98, dim=1)
            x_cat = torch.cat([xs[i // 2] if i % 2 == 0 else y for i in range(98*2)], dim=1)
        elif self.group == 196:
            xs = torch.chunk(x, 196, dim=1)
            x_cat = torch.cat([xs[i // 2] if i % 2 == 0 else y for i in range(392)], dim=1)
        elif self.group == 392:
            xs = torch.chunk(x, 392, dim=1)
            x_cat = torch.cat([xs[i // 2] if i % 2 == 0 else y for i in range(392*2)], dim=1)
        elif self.group == 784:
            xs = torch.chunk(x, 784, dim=1)
            x_cat = torch.cat([xs[i // 2] if i % 2 == 0 else y for i in range(784*2)], dim=1)
        elif self.group == 1568:
            xs = torch.chunk(x, 1568, dim=1)
            x_cat = torch.cat([xs[i // 2] if i % 2 == 0 else y for i in range(1568*2)], dim=1)
        else:
            print('group = {}'.format(self.group))
            raise Exception("Invalid Channel")

        x = x + self.conv(x_cat)
        y = y + self.score(x)

        return x, y


class ReverseStage(nn.Module):
    def __init__(self, channel):
        super(ReverseStage, self).__init__()
        self.weak_gra = GRA(channel, channel)
        self.medium_gra = GRA(channel, 8)
        self.strong_gra = GRA(channel, 1)

    def forward(self, x, y):
        # reverse guided block
        y = -1 * (torch.sigmoid(y)) + 1

        # three group-reversal attention blocks
        x, y = self.weak_gra(x, y)
        x, y = self.medium_gra(x, y)
        _, y = self.strong_gra(x, y)

        return y

#以粗糙预测图为中心，掩膜增强+补充高分辨率信息
class CentralReviseHeaBlk(nn.Module):
    def __init__(self, in_channels, k=12, kernel_size=3, radius=5):
        super(CentralReviseHeaBlk, self).__init__()
        self.k = k  # 复制倍率因子
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.radius = radius
        self.avg_pool = nn.AvgPool2d(kernel_size=(2 * radius + 1), stride=1, padding=radius)
        self.decoder1 = two_ConvBnRule(in_channels + 3 * k, in_channels + 3 * k)  # 更新输入通道数
        self.conv_out = nn.Conv2d(in_channels + 3 * k, in_channels, 1)
        self.conv_getguide = nn.Conv2d(in_channels, 1, 1)

    def forward(self, coarse_m1, raw_x):

        # Step 1: Apply a 1x1 convolution to reduce coarse_m1 to a single channel and apply softmax
        guide = self.conv_getguide(coarse_m1).sigmoid()
        coarse_softmax = guide  # Softmax along the channel dimension
        # Step 2: Calculate the maximum value at each position (no need to consider the channel dimension)
        # coarse_max_values = torch.max(coarse_softmax, dim=1, keepdim=True)[0]

        # Step 3: Create a mask to select positions where the maximum value is greater than a threshold
        mask = (coarse_softmax > 0.005).float()

        # Step 4: Apply average pooling to the mask to expand the region of interest
        expanded_mask = self.avg_pool(mask)
        expanded_mask = (expanded_mask > 0.09).float()
        expanded_mask = expanded_mask.expand(-1, 3, -1, -1)

        # Step 5: Resize raw_x to match the spatial dimensions of coarse_m1
        raw_x_resized = F.interpolate(raw_x, size=(coarse_m1.shape[2], coarse_m1.shape[3]), mode='bilinear', align_corners=False)


        # Step 6: Apply the mask to raw_x
        masked_raw_x = raw_x_resized * expanded_mask


        raw_x_resized_v = F.interpolate(masked_raw_x, size=(945,1200), mode='bilinear', align_corners=False)
        expanded_mask = F.interpolate(expanded_mask, size=(945,1200), mode='bilinear', align_corners=False)
        self.save_tensor_img(expanded_mask,'/home/amos/PycharmProjects/FD/visual_out/mask_1.jpg')
        self.save_tensor_img(raw_x_resized_v,'/home/amos/PycharmProjects/FD/visual_out/selected_1.jpg')

        # Step 7: Repeat masked_raw_x along the channel dimension
        masked_raw_x_repeated = masked_raw_x.repeat(1, self.k, 1, 1)

        # Step 7: Combine coarse_m1 with the masked raw_x
        combined_features = torch.cat([coarse_m1, masked_raw_x_repeated], dim=1)

        # Step 8: Perform convolution on the combined features

        fine_m = self.decoder1(combined_features)
        fine_m = self.conv_out(fine_m)

        return fine_m, guide

    def save_tensor_img(self, tensor, filename):
        # 将 tensor 移动到 CPU 并转为 numpy 数组
        tensor = tensor.detach().cpu()

        # 确保输入是三通道的
        if tensor.shape[1] != 3:
            raise ValueError("Input tensor must have 3 channels.")

        # 获取第一张图并转换为 (H, W, C) 格式
        mask_image = tensor[0].numpy()
        mask_image = np.transpose(mask_image, (1, 2, 0))

        # 确保值在 [0, 1] 范围内并转换为 [0, 255]
        mask_image = np.clip(mask_image * 255, 0, 255).astype('uint8')

        # 将 numpy 数组转换为 PIL 图像并保存
        mask_image = Image.fromarray(mask_image)
        mask_image.save(filename)