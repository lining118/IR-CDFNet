import logging
import os
import torch
from torchvision import transforms
import numpy as np
import random
import cv2
from PIL import Image
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import imageio





def path_to_image(path, size=(244, 244), color_type='rgb'):
    if color_type.lower() == 'rgb':
        image = cv2.imread(path)
    elif color_type.lower() == 'gray':
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    elif color_type.lower() == 'rgb2ycrcb':
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    else:
        print('Select the color_type to return, either rgb, gray, or rgb2ycrcb.')
        return

    if size:
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

    if color_type.lower() == 'rgb':
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert('RGB')
    elif color_type.lower() == 'gray':
        image = Image.fromarray(image).convert('L')
    elif color_type.lower() == 'rgb2ycrcb':
        image = Image.fromarray(image).convert('YCbCr')

    return image


def check_state_dict(state_dict, unwanted_prefix='_orig_mod.'):
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    return state_dict


def generate_smoothed_gt(gts):
    epsilon = 0.001
    new_gts = (1 - epsilon) * gts + epsilon / 2
    return new_gts


class Logger():
    def __init__(self, path="log.txt"):
        self.logger = logging.getLogger('SANet')
        self.file_handler = logging.FileHandler(path, "w")
        self.stdout_handler = logging.StreamHandler()
        self.stdout_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.stdout_handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

    def info(self, txt):
        self.logger.info(txt)

    def close(self):
        self.file_handler.close()
        self.stdout_handler.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, path, filename="latest.pth"):
    torch.save(state, os.path.join(path, filename))


def save_tensor_img(tenor_im, path):
    '''
    im = tenor_im.cpu().clone()
    im = im.squeeze(0)
    tensor2pil = transforms.ToPILImage()
    im = tensor2pil(im)
    # im = im.convert('L')
    im.save(path)
    '''
    # 将张量转换为numpy数组，并转换为uint8类型
    im = tenor_im.cpu().clone().squeeze().numpy().astype('uint8')    # 添加通道维度
    # print('im.shape = {}'.format(im.shape))
    # 保存图像
    imageio.imwrite(path, im)


def save_tensor_heatmap(tensor, path):
    # Convert tensor to numpy array
    array = tensor.cpu().detach().numpy().squeeze(0).squeeze(0)
    # Create heatmap
    plt.imshow(array, cmap='hot', interpolation='nearest')
    plt.axis('off')  # Turn off axis
    plt.colorbar()   # Add color bar
    plt.savefig(path, bbox_inches='tight', pad_inches=0.0)  # Save image
    plt.close()     # Close plot to release memory



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def save_tensor_img_for_latblk(tensor_im, save_dir):
    """
    将输入的 PyTorch tensor 批量可视化并保存为图像文件。

    参数:
        tensor_im (torch.Tensor): 输入的 PyTorch tensor，包含一批图像。
        save_dir (str): 要保存图像的目录路径。

    返回:
        无，将每张图像保存在指定目录下。
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 对每张图像进行遍历并保存
    for i in range(tensor_im.size(0)):
        # 获取单张图像的 tensor
        im_tensor = tensor_im[i].cpu().clone()

        # 将 tensor 转换为 PIL 图像
        image_pil = TF.to_pil_image(im_tensor.squeeze(0))

        # 保存图像
        image_pil.save(os.path.join(save_dir, f'image_{i}.png'))


def get_latblk_structure_weight(fds):
    [d1,d2,d3] = fds
    d1_ = F.interpolate(d1, size=244, mode='bilinear', align_corners=False)
    d2_ = F.interpolate(d2, size=244, mode='bilinear', align_corners=False)
    d3_ = F.interpolate(d3, size=244, mode='bilinear', align_corners=False)
    save_tensor_img_for_latblk(d1_, './testfolder/d1')
    save_tensor_img_for_latblk(d2_, './testfolder/d2')
    save_tensor_img_for_latblk(d3_, './testfolder/d3')


def Seg():
    dict = {0: 0, 1: 1, 2: 8, 3: 16, 4: 9, 5: 2, 6: 3, 7: 10, 8: 17,
                 9: 24, 10: 32, 11: 25, 12: 18, 13: 11, 14: 4, 15: 5, 16: 12,
                 17: 19, 18: 26, 19: 33, 20: 40, 21: 48, 22: 41, 23: 34, 24: 27,
                 25: 20, 26: 13, 27: 6, 28: 7, 29: 14, 30: 21, 31: 28, 32: 35,
                 33: 42, 34: 49, 35: 56, 36: 57, 37: 50, 38: 43, 39: 36, 40: 29,
                 41: 22, 42: 15, 43: 23, 44: 30, 45: 37, 46: 44, 47: 51, 48: 58,
                 49: 59, 50: 52, 51: 45, 52: 38, 53: 31, 54: 39, 55: 46, 56: 53,
                 57: 60, 58: 61, 59: 54, 60: 47, 61: 55, 62: 62, 63: 63}
    a = torch.zeros(1, 64, 1, 1)

    for i in range(0, 32):
        a[0, dict[i+32], 0, 0] = 1

    return a

def norm(x):
    return (1 - torch.exp(-x)) / (1 + torch.exp(-x))  #存在溢出？
    # return torch.tanh(x)