# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch.utils.data
from .torchvision_datasets import CocoDetection

from .coco import build as build_coco


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset#如果是子集，那么返回原数据集
    if isinstance(dataset, CocoDetection):
        return dataset.coco#如果是原数据集，即TvCocoDetection子类，那么返回COCO类，包含各种数据集的索引属性


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':#默认为coco数据集，记得按该数据集调整文件夹结构和注释文件结构
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':#暂不知coco数据集文件夹结构，估计不需要该数据集，故略过
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
