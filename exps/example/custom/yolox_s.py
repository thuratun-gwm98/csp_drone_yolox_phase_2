#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "./datasets/HIT_UAV_NormalJson_dataset/"
        self.train_ann = "train.json"
        self.val_ann = "val.json"

        self.output_dir = "./YOLOX_outputs/hit_uav_150_eph"

        self.num_classes = 4

        self.max_epoch = 150
        self.data_num_workers = 2

        self.input_size = (512, 512)
        self.test_size = (512, 512)

        self.eval_interval = 5
        self.print_interval = 5

        self.save_history_ckpt = False

        # hyperparameter
        self.min_lr_ratio = 0.05
        self.weight_decay = 0.0002
        self.momentum = 0.9

        # Augmentation Config
        self.degrees = 15.0
        self.shear = 3.0
        self.mosaic_scale = (0.1, 0.9)

        self.enable_mixup = True
        self.mixup_scale = (0.5, 1.5)
