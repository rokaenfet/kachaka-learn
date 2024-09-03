from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import numpy as np

import cv2
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import sys
from tensorboardX import SummaryWriter

import _init_paths
from lib.config import cfg
from lib.config import update_config

from lib.core.loss import JointsMSELoss
from lib.core.loss import DepthLoss
from lib.core.loss import hoe_diff_loss
from lib.core.loss import Bone_loss

from lib.core.function import train
from lib.core.function import validate

from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger
from lib.utils.utils import get_model_summary

import lib.dataset
from lib.models.pose_hrnet import get_pose_net
from PIL import Image

class MEBOWFrame():
    def __init__(self):
        parser = argparse.ArgumentParser(description='Train keypoints network')
        parser.add_argument('--cfg',
                            help='experiment configure file name',
                            type=str,
                            default="MEBOW/experiments/coco/segm-4_lr1e-3.yaml")

        parser.add_argument('opts',
                            help="Modify config options using the command-line",
                            default=None,
                            nargs=argparse.REMAINDER)

        # philly
        parser.add_argument('--modelDir',
                            help='model directory',
                            type=str,
                            default='')
        parser.add_argument('--logDir',
                            help='log directory',
                            type=str,
                            default='')
        parser.add_argument('--dataDir',
                            help='data directory',
                            type=str,
                            default='')
        parser.add_argument('--prevModelDir',
                            help='prev Model directory',
                            type=str,
                            default='')
        parser.add_argument('--device', default='cpu')
        # parser.add_argument('img_path')
        args = parser.parse_args()

        update_config(cfg, args)

        # logger, _, _ = create_logger(
        #     cfg, args.cfg, 'valid')

        # logger.info(pprint.pformat(args))
        # logger.info(cfg)

        # cudnn related setting
        cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

        self.model = get_pose_net(cfg, is_train=False).to(args.device)

        print(f"\033[32mSuccessfully\033[0m imported stalk/models/model_hboe.pth")
        self.model.load_state_dict(torch.load("stalk/models/model_hboe.pth", map_location=torch.device(args.device)), strict=True)

        # Data loading code
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.transform = transforms.Compose([
                transforms.ToTensor(),
                normalize])
        
    async def process(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (192, 256))
        input = self.transform(img).unsqueeze(0)
        input = input.float()

        self.model.eval()
        _, self.hoe_output = self.model(input)
        self.ori = torch.argmax(self.hoe_output[0]) * 5