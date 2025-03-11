import os
import torch
import math


class Config:
    def __init__(self):
        # get abs_dir

        self.abs_file = os.path.abspath(__file__)
        self.abs_dir = self.abs_file[:self.abs_file.rfind('\\')] if os.name == 'nt' else self.abs_file[
                                                                                         :self.abs_file.rfind(r'/')]

        # ========= Manually Setup=========#

        self.task = ['COD'][0]
        self.train_method = ['local', 'cloud'][0]
        self.backbone = [
            'vgg16', 'vgg16bn', 'resnet50',  # 0, 1, 2
            'pvt_v2_b2', 'pvt_v2_b5',  # 3-bs10, 4-bs5
            'swin_v1_b', 'swin_v1_l',  # 5-bs9, 6-bs6
            'swin_v1_t', 'swin_v1_s',  # 7, 8
            'pvt_v2_b0', 'pvt_v2_b1',  # 9, 10
            'MambaVision_b_1k', 'MambaVision_l_1k'  # https://github.com/NVlabs/MambaVision 11,12
        ][6]
        self.backbone_weights_file_name = {
            'pvt_v2_b2': 'pvt_v2_b2.pth',
            'pvt_v2_b5': ['pvt_v2_b5.pth', 'pvt_v2_b5_22k.pth'][0],
            'swin_v1_b': ['swin_base_patch4_window12_384_22kto1k.pth', 'swin_base_patch4_window12_384_22k.pth'][0],
            'swin_v1_l': ['swin_large_patch4_window12_384_22kto1k.pth', 'swin_large_patch4_window12_384_22k.pth'][0],
            'swin_v1_t': ['swin_tiny_patch4_window7_224_22kto1k_finetune.pth'][0],
            'swin_v1_s': ['swin_small_patch4_window7_224_22kto1k_finetune.pth'][0],
            'pvt_v2_b0': ['pvt_v2_b0.pth'][0],
            'pvt_v2_b1': ['pvt_v2_b1.pth'][0],
            'MambaVision_b_1k': ['online'][0],
            'MambaVision_l_1k': ['online'][0],
        }[self.backbone]
        self.is_pretrained_backbone = [True, False][0]
        self.mul_scl_ipt = ['none', 'cat', 'add'][2]
        self.mul_scl_sc = [True, False][1]  # multi scl skip connection
        self.mul_scl_sc_num = [0, 3][1]
        self.mul_lev_ipt = False
        if not self.mul_scl_sc:
            self.mul_lev_ipt = [True, False][1]  # low, middle, high level
        self.channels_list = [3, 192, 384, 768, 1536]   # [3, 192, 384, 768, 1536] swin L #mamba[3, 196, 392, 784, 1568]
        self.dec_target_size = [8, 16, 31, 61]
        self.optimizer = ['Adam', 'AdamW'][0]
        self.lr_schedule = ['StepLR', 'MultiStepLR', 'ExponentialLR', 'LinearLR', 'CyclicLR', 'OneCycleLR',
                            'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'LambdaLR', 'SequentialLR',
                            'ChainedScheduler', 'ConstantLR', 'ReduceLROnPlateau'][3]
        # self.preproc_methods = ['flip', 'enhance', 'rotate', 'pepper', 'crop'][:4]
        self.train_size = [244, 256, 512, 352][2]
        self.att_blk = ['SpatialAttention', 'ChannelAttention', 'MixResAttention','HOR'][3]
        self.dec_blk = ['BasicDecBlk', 'RF'][0]
        self.lat_blk = ['GussianLatBlk','two_ConvBnRule'][1]
        self.lat_blk_filter = [False, True][0]
        self.hea_blk = ['ReverseStage','CentralReviseHeaBlk'][1]
        self.mix_blk = ['PAM','DAC','FRCv2'][1]
        self.enh_blk = ['FSF'][0]
        self.enh_blk2 = ['FEM2'][0]
        self.out_ref = [True, False][1]
        self.only_S_MAE = [True, False][1]
        self.IoU_finetune_last_epochs = [0, -20][1]     # choose 0 to skip
        self.load_all = True
        self.auxiliary_classification = False
        # filter configs
        self.gus_ker_type = ['2d', '3d'][0]  # build 2d gussian kernal or 3d
        self.verbose_eval = False
        # self.train_notice = [False, True][1]






        # ========= Automatically Configs =========#

        # Model Configs
        self.model = {'COD': 'SANet'}[self.task]
        self.backbone_weights_root_dir = 'lib/weights/backbones'
        self.backbone_weights_dir = os.path.join(self.abs_dir, self.backbone_weights_root_dir,
                                                 self.backbone_weights_file_name)

        # Train Configs
        self.resume = True
        self.batch_size = {
            'local': 4,
            'cloud': 32
        }[self.train_method]
        self.num_workers = 6  # will be decrease to min(it, batch_size) at the initialization of the data_loader
        self.lr = 1e-5 * math.sqrt(self.batch_size / 5)  # adapt the lr linearly
        self.lr_decay_epochs = [1e4]    # Set to negative N to decay the lr in the last N-th epoch.
        self.lr_decay_rate = 0.5
        self.lambdas_pix_last = {
            # not 0 means opening this loss
            # original rate -- 1 : 30 : 1.5 : 0.2, bce x 30
            'bce': 1 * 30,          # high performance
            'iou': 0.5 * 1,         # 0 / 255
            'iou_patch': 0.5 * 0,   # 0 / 255, win_size = (64, 64)
            'mse': 150 * 0,         # can smooth the saliency map
            'triplet': 3 * 0,
            'reg': 100 * 0,
            'ssim': 10 * 1,          # help contours,
            'cnt': 5 * 0,          # help contours
        }

        # Data Configs
        self.dataset_root = os.path.join(self.abs_dir, 'data')
        self.training_set = {
            'COD': 'COD10K_CAMO_CHAMELEON_NC4K_TrainingDataset'
        }[self.task]
        self.training_set_root_dir = 'data/train_dataset'
        self.training_set_dir = os.path.join(self.abs_dir, self.training_set_root_dir,
                                             self.training_set)

        # others
        self.device = [0, 'cpu'][0 if torch.cuda.is_available() else 1]     # .to(0) == .to('cuda:0')

        self.batch_size_valid = 1
        self.rand_seed = 7
        run_sh_file = [f for f in os.listdir('.') if 'train.sh' == f] + [os.path.join('..', f) for f in os.listdir('..') if 'train.sh' == f]
        with open(run_sh_file[0], 'r') as f:
            lines = f.readlines()
            self.save_last = int([l.strip() for l in lines if 'val_last=' in l][0].split('=')[-1])
            self.save_step = int([l.strip() for l in lines if 'step=' in l][0].split('=')[-1])
        self.val_step = [0, self.save_step][0]

'''
test = Config()

print(test.lambdas_pix_last)
'''
