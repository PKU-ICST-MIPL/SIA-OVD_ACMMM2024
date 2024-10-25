# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch
import torch.nn.functional as F

import util.misc as utils
from datasets.coco_eval import CocoEvaluator, convert_to_xywh
from datasets.panoptic_eval import PanopticEvaluator
from models.fast_detr import contrastive_loss
import torchvision
from util.box_ops import box_cxcywh_to_xyxy
from torch.nn.functional import cross_entropy

# adapted from dab-detr
def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos

def prepare_adapter_output(model, output_feats, targets, num_adapters):
    assert num_adapters in [1,2,4,10,20]
    if num_adapters == 1:
        if 'module' in dir(model):
            adapter_output_1 = model.module.shape_adapter_1(output_feats)
        else:
            adapter_output_1 = model.shape_adapter_1(output_feats)
        x = adapter_output_1

        ratio = 0.2
        output_feats = ratio * x + (1 - ratio) * output_feats
    elif num_adapters == 2:
        if 'module' in dir(model):
            # adapter_output_xiaoyu1 = model.module.shape_adapter_xiaoyu1(output_feats)
            # adapter_output_dayu1 = model.module.shape_adapter_dayu1(output_feats)
            adapter_output_1 = model.module.shape_adapter_1(output_feats)
            adapter_output_2 = model.module.shape_adapter_2(output_feats)
        else:
            # adapter_output_xiaoyu1 = model.shape_adapter_xiaoyu1(output_feats)
            # adapter_output_dayu1 = model.shape_adapter_dayu1(output_feats)
            adapter_output_1 = model.shape_adapter_1(output_feats)
            adapter_output_2 = model.shape_adapter_2(output_feats)

        which_adapter = [] # ['1', '2', '1', '2', '2', '1', '2', '1']
        for target in targets:
            for box in target['ori_boxes']:
                w = box[2]
                h = box[3]
                if w/h <= 1/1:
                    which_adapter.append('1')
                elif 1/1< w/h:
                    which_adapter.append('2')

        adapter_outputs = []
        for adapter_num, tensor_1, tensor_2 in zip(which_adapter, adapter_output_1, adapter_output_2):
            # 根据适配器选择线性层
            if adapter_num == '1':
                adapter_output = tensor_1 * 1 + tensor_2 * 0               
                adapter_outputs.append(adapter_output)
            elif adapter_num == '2':
                adapter_output = tensor_1 * 0 + tensor_2 * 1
                adapter_outputs.append(adapter_output)
            else:
                raise ValueError("Invalid adapter value")
        x = torch.stack(adapter_outputs, dim=0)

        ratio = 0.2
        output_feats = ratio * x + (1 - ratio) * output_feats
    elif num_adapters == 4:
        if 'module' in dir(model):
            # adapter_output_xiaoyu1 = model.module.shape_adapter_xiaoyu1(output_feats)
            # adapter_output_dayu1 = model.module.shape_adapter_dayu1(output_feats)
            adapter_output_1 = model.module.shape_adapter_1(output_feats)
            adapter_output_2 = model.module.shape_adapter_2(output_feats)
            adapter_output_3 = model.module.shape_adapter_3(output_feats)
            adapter_output_4 = model.module.shape_adapter_4(output_feats)
        else:
            # adapter_output_xiaoyu1 = model.shape_adapter_xiaoyu1(output_feats)
            # adapter_output_dayu1 = model.shape_adapter_dayu1(output_feats)
            adapter_output_1 = model.shape_adapter_1(output_feats)
            adapter_output_2 = model.shape_adapter_2(output_feats)
            adapter_output_3 = model.shape_adapter_3(output_feats)
            adapter_output_4 = model.shape_adapter_4(output_feats)

        which_adapter = [] # ['1', '2', '1', '2', '2', '1', '2', '1']
        for target in targets:
            for box in target['ori_boxes']:
                w = box[2]
                h = box[3]
                if w/h <= 1/6:
                    which_adapter.append('1')
                elif 1/6< w/h <= 1/1:
                    which_adapter.append('2')
                elif 1/1< w/h <= 6/1:
                    which_adapter.append('3')
                elif 6/1< w/h:
                    which_adapter.append('4')

        adapter_outputs = []
        for adapter_num, tensor_1, tensor_2, tensor_3, tensor_4 in zip(which_adapter, adapter_output_1, adapter_output_2, adapter_output_3, adapter_output_4):
            # 根据适配器选择线性层
            if adapter_num == '1':
                adapter_output = tensor_1 * 1 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0                 
                adapter_outputs.append(adapter_output)
            elif adapter_num == '2':
                adapter_output = tensor_1 * 0 + tensor_2 * 1 + tensor_3 * 0 + tensor_4 * 0 
                adapter_outputs.append(adapter_output)
            elif adapter_num == '3':
                adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 1 + tensor_4 * 0 
                adapter_outputs.append(adapter_output)
            elif adapter_num == '4':
                adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 1 
                adapter_outputs.append(adapter_output)
            else:
                raise ValueError("Invalid adapter value")
        x = torch.stack(adapter_outputs, dim=0)

        ratio = 0.2
        output_feats = ratio * x + (1 - ratio) * output_feats
    elif num_adapters == 10:
        if 'module' in dir(model):
            # adapter_output_xiaoyu1 = model.module.shape_adapter_xiaoyu1(output_feats)
            # adapter_output_dayu1 = model.module.shape_adapter_dayu1(output_feats)
            adapter_output_1 = model.module.shape_adapter_1(output_feats)
            adapter_output_2 = model.module.shape_adapter_2(output_feats)
            adapter_output_3 = model.module.shape_adapter_3(output_feats)
            adapter_output_4 = model.module.shape_adapter_4(output_feats)
            adapter_output_5 = model.module.shape_adapter_5(output_feats)
            adapter_output_6 = model.module.shape_adapter_6(output_feats)
            adapter_output_7 = model.module.shape_adapter_7(output_feats)
            adapter_output_8 = model.module.shape_adapter_8(output_feats)
            adapter_output_9 = model.module.shape_adapter_9(output_feats)
            adapter_output_10 = model.module.shape_adapter_10(output_feats)
        else:
            # adapter_output_xiaoyu1 = model.shape_adapter_xiaoyu1(output_feats)
            # adapter_output_dayu1 = model.shape_adapter_dayu1(output_feats)
            adapter_output_1 = model.shape_adapter_1(output_feats)
            adapter_output_2 = model.shape_adapter_2(output_feats)
            adapter_output_3 = model.shape_adapter_3(output_feats)
            adapter_output_4 = model.shape_adapter_4(output_feats)
            adapter_output_5 = model.shape_adapter_5(output_feats)
            adapter_output_6 = model.shape_adapter_6(output_feats)
            adapter_output_7 = model.shape_adapter_7(output_feats)
            adapter_output_8 = model.shape_adapter_8(output_feats)
            adapter_output_9 = model.shape_adapter_9(output_feats)
            adapter_output_10 = model.shape_adapter_10(output_feats)
        

        which_adapter = [] # ['1', '2', '1', '2', '2', '1', '2', '1']
        for target in targets:
            for box in target['ori_boxes']:
                w = box[2]
                h = box[3]
                if w/h <= 1/9:
                    which_adapter.append('1')
                elif 1/9< w/h <= 1/7:
                    which_adapter.append('2')
                elif 1/7< w/h <= 1/5:
                    which_adapter.append('3')
                elif 1/5< w/h <= 1/3:
                    which_adapter.append('4')
                elif 1/3< w/h <= 1/1:
                    which_adapter.append('5')
                elif 1/1< w/h <= 3/1:
                    which_adapter.append('6')
                elif 3/1< w/h <= 5/1:
                    which_adapter.append('7')
                elif 5/1< w/h <= 7/1:
                    which_adapter.append('8')
                elif 7/1< w/h <= 9/1:
                    which_adapter.append('9')
                elif 9/1< w/h:
                    which_adapter.append('10')
                else:
                    raise ValueError("Invalid adapter value")

        adapter_outputs = []
        for adapter_num, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7, tensor_8, tensor_9, tensor_10, \
            in zip(which_adapter, adapter_output_1, adapter_output_2, adapter_output_3, adapter_output_4, adapter_output_5, \
                   adapter_output_6, adapter_output_7, adapter_output_8, adapter_output_9, adapter_output_10
                    ):
            # 根据适配器选择线性层
            if adapter_num == '1':
                adapter_output = tensor_1 * 1 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0
                adapter_outputs.append(adapter_output)
            elif adapter_num == '2':
                adapter_output = tensor_1 * 0 + tensor_2 * 1 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0
                adapter_outputs.append(adapter_output)
            elif adapter_num == '3':
                adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 1 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0
                adapter_outputs.append(adapter_output)
            elif adapter_num == '4':
                adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 1 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0
                adapter_outputs.append(adapter_output)
            elif adapter_num == '5':
                adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 1 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0
                adapter_outputs.append(adapter_output)
            elif adapter_num == '6':
                adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 1 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 
                adapter_outputs.append(adapter_output)
            elif adapter_num == '7':
                adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 1 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0
                adapter_outputs.append(adapter_output)
            elif adapter_num == '8':
                adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 1 + tensor_9 * 0 + tensor_10 * 0
                adapter_outputs.append(adapter_output)
            elif adapter_num == '9':
                adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 1 + tensor_10 * 0
                adapter_outputs.append(adapter_output)
            elif adapter_num == '10':
                adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 1
                adapter_outputs.append(adapter_output)
            else:
                raise ValueError("Invalid adapter value")
        x = torch.stack(adapter_outputs, dim=0)

        ratio = 0.2
        output_feats = ratio * x + (1 - ratio) * output_feats
    elif num_adapters == 20:
        if 'module' in dir(model):
            # adapter_output_xiaoyu1 = model.module.shape_adapter_xiaoyu1(output_feats)
            # adapter_output_dayu1 = model.module.shape_adapter_dayu1(output_feats)
            adapter_output_1 = model.module.shape_adapter_1(output_feats)
            adapter_output_2 = model.module.shape_adapter_2(output_feats)
            adapter_output_3 = model.module.shape_adapter_3(output_feats)
            adapter_output_4 = model.module.shape_adapter_4(output_feats)
            adapter_output_5 = model.module.shape_adapter_5(output_feats)
            adapter_output_6 = model.module.shape_adapter_6(output_feats)
            adapter_output_7 = model.module.shape_adapter_7(output_feats)
            adapter_output_8 = model.module.shape_adapter_8(output_feats)
            adapter_output_9 = model.module.shape_adapter_9(output_feats)
            adapter_output_10 = model.module.shape_adapter_10(output_feats)
            adapter_output_11 = model.module.shape_adapter_11(output_feats)
            adapter_output_12 = model.module.shape_adapter_12(output_feats)
            adapter_output_13 = model.module.shape_adapter_13(output_feats)
            adapter_output_14 = model.module.shape_adapter_14(output_feats)
            adapter_output_15 = model.module.shape_adapter_15(output_feats)
            adapter_output_16 = model.module.shape_adapter_16(output_feats)
            adapter_output_17 = model.module.shape_adapter_17(output_feats)
            adapter_output_18 = model.module.shape_adapter_18(output_feats)
            adapter_output_19 = model.module.shape_adapter_19(output_feats)
            adapter_output_20 = model.module.shape_adapter_20(output_feats)
        else:
            # adapter_output_xiaoyu1 = model.shape_adapter_xiaoyu1(output_feats)
            # adapter_output_dayu1 = model.shape_adapter_dayu1(output_feats)
            adapter_output_1 = model.shape_adapter_1(output_feats)
            adapter_output_2 = model.shape_adapter_2(output_feats)
            adapter_output_3 = model.shape_adapter_3(output_feats)
            adapter_output_4 = model.shape_adapter_4(output_feats)
            adapter_output_5 = model.shape_adapter_5(output_feats)
            adapter_output_6 = model.shape_adapter_6(output_feats)
            adapter_output_7 = model.shape_adapter_7(output_feats)
            adapter_output_8 = model.shape_adapter_8(output_feats)
            adapter_output_9 = model.shape_adapter_9(output_feats)
            adapter_output_10 = model.shape_adapter_10(output_feats)
            adapter_output_11 = model.shape_adapter_11(output_feats)
            adapter_output_12 = model.shape_adapter_12(output_feats)
            adapter_output_13 = model.shape_adapter_13(output_feats)
            adapter_output_14 = model.shape_adapter_14(output_feats)
            adapter_output_15 = model.shape_adapter_15(output_feats)
            adapter_output_16 = model.shape_adapter_16(output_feats)
            adapter_output_17 = model.shape_adapter_17(output_feats)
            adapter_output_18 = model.shape_adapter_18(output_feats)
            adapter_output_19 = model.shape_adapter_19(output_feats)
            adapter_output_20 = model.shape_adapter_20(output_feats)

        which_adapter = [] # ['1', '2', '1', '2', '2', '1', '2', '1']
        for target in targets:
            for box in target['ori_boxes']:
                w = box[2]
                h = box[3]
                if w/h <= 1/10:
                    which_adapter.append('1')
                elif 1/10< w/h <= 1/9:
                    which_adapter.append('2')
                elif 1/9< w/h <= 1/8:
                    which_adapter.append('3')
                elif 1/8< w/h <= 1/7:
                    which_adapter.append('4')
                elif 1/7< w/h <= 1/6:
                    which_adapter.append('5')
                elif 1/6< w/h <= 1/5:
                    which_adapter.append('6')
                elif 1/5< w/h <= 1/4:
                    which_adapter.append('7')
                elif 1/4< w/h <= 1/3:
                    which_adapter.append('8')
                elif 1/3< w/h <= 1/2:
                    which_adapter.append('9')
                elif 1/2< w/h <= 1/1:
                    which_adapter.append('10')
                elif 1/1< w/h <= 2/1:
                    which_adapter.append('11')
                elif 2/1< w/h <= 3/1:
                    which_adapter.append('12')
                elif 3/1< w/h <= 4/1:
                    which_adapter.append('13')
                elif 4/1< w/h <= 5/1:
                    which_adapter.append('14')
                elif 5/1< w/h <= 6/1:
                    which_adapter.append('15')
                elif 6/1< w/h <= 7/1:
                    which_adapter.append('16')
                elif 7/1< w/h <= 8/1:
                    which_adapter.append('17')
                elif 8/1< w/h <= 9/1:
                    which_adapter.append('18')
                elif 9/1< w/h <= 10/1:
                    which_adapter.append('19')
                elif 10/1< w/h:
                    which_adapter.append('20')

        adapter_outputs = []
        for adapter_num, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7, tensor_8, tensor_9, tensor_10, \
            tensor_11, tensor_12, tensor_13, tensor_14, tensor_15, tensor_16, tensor_17, tensor_18, tensor_19, tensor_20 \
            in zip(which_adapter, adapter_output_1, adapter_output_2, adapter_output_3, adapter_output_4, adapter_output_5, \
                   adapter_output_6, adapter_output_7, adapter_output_8, adapter_output_9, adapter_output_10, \
                   adapter_output_11, adapter_output_12, adapter_output_13, adapter_output_14, adapter_output_15, \
                   adapter_output_16, adapter_output_17, adapter_output_18, adapter_output_19, adapter_output_20,
                    ):
            # 根据适配器选择线性层
            if adapter_num == '1':
                adapter_output = tensor_1 * 1 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 + \
                                 tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 0 
                adapter_outputs.append(adapter_output)
            elif adapter_num == '2':
                adapter_output = tensor_1 * 0 + tensor_2 * 1 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 + \
                                 tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 0
                adapter_outputs.append(adapter_output)
            elif adapter_num == '3':
                adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 1 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 + \
                                 tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 0
                adapter_outputs.append(adapter_output)
            elif adapter_num == '4':
                adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 1 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 + \
                                 tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 0
                adapter_outputs.append(adapter_output)
            elif adapter_num == '5':
                adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 1 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 + \
                                 tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 0
                adapter_outputs.append(adapter_output)
            elif adapter_num == '6':
                adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 1 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 + \
                                 tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 0
                adapter_outputs.append(adapter_output)
            elif adapter_num == '7':
                adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 1 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 + \
                                 tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 0
                adapter_outputs.append(adapter_output)
            elif adapter_num == '8':
                adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 1 + tensor_9 * 0 + tensor_10 * 0 + \
                                 tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 0
                adapter_outputs.append(adapter_output)
            elif adapter_num == '9':
                adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 1 + tensor_10 * 0 + \
                                 tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 0
                adapter_outputs.append(adapter_output)
            elif adapter_num == '10':
                adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 1 + \
                                 tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 0
                adapter_outputs.append(adapter_output)
            elif adapter_num == '11':
                adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 + \
                                 tensor_11 * 1 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 0 
                adapter_outputs.append(adapter_output)
            elif adapter_num == '12':
                adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 + \
                                 tensor_11 * 0 + tensor_12 * 1 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 0
                adapter_outputs.append(adapter_output)
            elif adapter_num == '13':
                adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 + \
                                 tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 1 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 0
                adapter_outputs.append(adapter_output)
            elif adapter_num == '14':
                adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 + \
                                 tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 1 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 0
                adapter_outputs.append(adapter_output)
            elif adapter_num == '15':
                adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 + \
                                 tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 1 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 0
                adapter_outputs.append(adapter_output)
            elif adapter_num == '16':
                adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 + \
                                 tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 1 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 0
                adapter_outputs.append(adapter_output)
            elif adapter_num == '17':
                adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 + \
                                 tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 1 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 0
                adapter_outputs.append(adapter_output)
            elif adapter_num == '18':
                adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 + \
                                 tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 1 + tensor_19 * 0 + tensor_20 * 0
                adapter_outputs.append(adapter_output)
            elif adapter_num == '19':
                adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 + \
                                 tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 1 + tensor_20 * 0
                adapter_outputs.append(adapter_output)
            elif adapter_num == '20':
                adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 + \
                                 tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 1
                adapter_outputs.append(adapter_output)
            else:
                raise ValueError("Invalid adapter value")
        x = torch.stack(adapter_outputs, dim=0)

        ratio = 0.2
        output_feats = ratio * x + (1 - ratio) * output_feats

    return output_feats

def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    max_norm: float = 0,
                    args=None,
                    accumulation_steps: int = 2):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    _cnt = 0
    for samples, targets, iteration in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v if isinstance(v, (list, dict)) else v.to(device) for k, v in t.items()} for t in targets]

        categories = data_loader.dataset.category_list

        # add pseudo labels
        pseudo_categories = list(set([a for target in targets if 'pseudo_labels' in target for a in target['pseudo_labels']]))
        for target in targets:
            if 'pseudo_labels' not in target:
                continue
            pseudo_label_ids = [pseudo_categories.index(cat) + len(categories) for cat in target['pseudo_labels']]
            target['labels'] = torch.cat([target['labels'], torch.tensor(pseudo_label_ids, device=target['labels'].device, dtype=target['labels'].dtype)])

        outputs = model(samples, categories=categories + pseudo_categories)
        
        features, text_feature, tau = outputs['features'], outputs['text_feature'], outputs['tau']
        
        if args.box_conditioned_pe:
            xywh_gt = torch.cat([target['boxes'] for target in targets])
            box_emb = gen_sineembed_for_position(xywh_gt.unsqueeze(0))[0]
            if args.only_box_size:
                box_emb = box_emb[:,256:]
        else:
            box_emb = None
        gt_boxes = [box_cxcywh_to_xyxy(target['boxes']) for target in targets]
        masks = features[0].decompose()[1]
        sizes = [((1 - m[0].float()).sum(), (1 - m[:,0].float()).sum()) for m in masks]
        for i in range(len(gt_boxes)):
            gt_boxes[i][:,[0,2]] = gt_boxes[i][:,[0,2]] * sizes[i][0]
            gt_boxes[i][:,[1,3]] = gt_boxes[i][:,[1,3]] * sizes[i][1]
        
        if args.roi_feat == 'layer4':
            if args.backbone == 'clip_RN50x4':
                reso = 9
            else:
                reso = 7
            roi_features = torchvision.ops.roi_align(
                features[0].tensors,
                gt_boxes,
                output_size=(reso, reso),
                spatial_scale=1.0,
                aligned=True)  # (bs * num_queries, c, 7, 7)
        
            if 'module' in dir(model):
                output_feats = model.module.backbone[0].attnpool(roi_features, box_emb)
            else:
                output_feats = model.backbone[0].attnpool(roi_features, box_emb)
                
        elif args.roi_feat == 'layer3':
            if args.backbone == 'clip_RN50x4':
                reso = 18
            else:
                reso = 14
            roi_features = torchvision.ops.roi_align(
                features[0].tensors,
                gt_boxes, # 4,n,4
                output_size=(reso, reso),
                spatial_scale=1.0,
                aligned=True)  # (bs * num_queries, c, 7, 7) 8,1024,14,14
        
            if 'module' in dir(model):
                output_feats = model.module.backbone[0].layer4(roi_features)
                output_feats = model.module.backbone[0].attnpool(output_feats, box_emb)
            else:
                output_feats = model.backbone[0].layer4(roi_features)
                output_feats = model.backbone[0].attnpool(output_feats, box_emb) # 8,1024 torch.Size([8, 1024]) bs,640

        #********************************************** adapter
        # if 'module' in dir(model):
        #     x = model.module.shape_adapter(output_feats)
        # else:
        #     x = model.shape_adapter(output_feats)
        # ratio = 0.2
        # output_feats = ratio * x + (1 - ratio) * output_feats

        # if 'module' in dir(model):
        #     # adapter_output_xiaoyu1 = model.module.shape_adapter_xiaoyu1(output_feats)
        #     # adapter_output_dayu1 = model.module.shape_adapter_dayu1(output_feats)
        #     adapter_output_1 = model.module.shape_adapter_1(output_feats)
        #     adapter_output_2 = model.module.shape_adapter_2(output_feats)
        #     adapter_output_3 = model.module.shape_adapter_3(output_feats)
        #     adapter_output_4 = model.module.shape_adapter_4(output_feats)
        #     adapter_output_5 = model.module.shape_adapter_5(output_feats)
        #     adapter_output_6 = model.module.shape_adapter_6(output_feats)
        #     adapter_output_7 = model.module.shape_adapter_7(output_feats)
        #     adapter_output_8 = model.module.shape_adapter_8(output_feats)
        #     adapter_output_9 = model.module.shape_adapter_9(output_feats)
        #     adapter_output_10 = model.module.shape_adapter_10(output_feats)
        # else:
        #     # adapter_output_xiaoyu1 = model.shape_adapter_xiaoyu1(output_feats)
        #     # adapter_output_dayu1 = model.shape_adapter_dayu1(output_feats)
        #     adapter_output_1 = model.shape_adapter_1(output_feats)
        #     adapter_output_2 = model.shape_adapter_2(output_feats)
        #     adapter_output_3 = model.shape_adapter_3(output_feats)
        #     adapter_output_4 = model.shape_adapter_4(output_feats)
        #     adapter_output_5 = model.shape_adapter_5(output_feats)
        #     adapter_output_6 = model.shape_adapter_6(output_feats)
        #     adapter_output_7 = model.shape_adapter_7(output_feats)
        #     adapter_output_8 = model.shape_adapter_8(output_feats)
        #     adapter_output_9 = model.shape_adapter_9(output_feats)
        #     adapter_output_10 = model.shape_adapter_10(output_feats)

        # which_adapter = [] # ['1', '2', '1', '2', '2', '1', '2', '1']
        # for target in targets:
        #     for box in target['ori_boxes']:
        #         w = box[2]
        #         h = box[3]
        #         if w/h <= 1/10:
        #             which_adapter.append('1')
        #         elif 1/10< w/h <= 1/8:
        #             which_adapter.append('2')
        #         elif 1/8< w/h <= 0.529:
        #             which_adapter.append('3')
        #         elif 0.529< w/h <= 0.653:
        #             which_adapter.append('4')
        #         elif 0.653< w/h <= 0.8:
        #             which_adapter.append('5')
        #         elif 0.8< w/h <= 0.98:
        #             which_adapter.append('6')
        #         elif 0.98< w/h <= 1.2:
        #             which_adapter.append('7')
        #         elif 1.2< w/h <= 1.52:
        #             which_adapter.append('8')
        #         elif 1.52< w/h <= 2.19:
        #             which_adapter.append('9')
        #         elif 2.19< w/h:
        #             which_adapter.append('10')

        # adapter_outputs = []
        # for adapter_num, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7, tensor_8, tensor_9, tensor_10, \
        #     in zip(which_adapter, adapter_output_1, adapter_output_2, adapter_output_3, adapter_output_4, adapter_output_5, \
        #            adapter_output_6, adapter_output_7, adapter_output_8, adapter_output_9, adapter_output_10
        #             ):
        #     # 根据适配器选择线性层
        #     if adapter_num == '1':
        #         adapter_output = tensor_1 * 1 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0
        #         adapter_outputs.append(adapter_output)
        #     elif adapter_num == '2':
        #         adapter_output = tensor_1 * 0 + tensor_2 * 1 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0
        #         adapter_outputs.append(adapter_output)
        #     elif adapter_num == '3':
        #         adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 1 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0
        #         adapter_outputs.append(adapter_output)
        #     elif adapter_num == '4':
        #         adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 1 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0
        #         adapter_outputs.append(adapter_output)
        #     elif adapter_num == '5':
        #         adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 1 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0
        #         adapter_outputs.append(adapter_output)
        #     elif adapter_num == '6':
        #         adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 1 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 
        #         adapter_outputs.append(adapter_output)
        #     elif adapter_num == '7':
        #         adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 1 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0
        #         adapter_outputs.append(adapter_output)
        #     elif adapter_num == '8':
        #         adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 1 + tensor_9 * 0 + tensor_10 * 0
        #         adapter_outputs.append(adapter_output)
        #     elif adapter_num == '9':
        #         adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 1 + tensor_10 * 0
        #         adapter_outputs.append(adapter_output)
        #     elif adapter_num == '10':
        #         adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 1
        #         adapter_outputs.append(adapter_output)
        #     else:
        #         raise ValueError("Invalid adapter value")
        # x = torch.stack(adapter_outputs, dim=0)

        # ratio = 0.2
        # output_feats = ratio * x + (1 - ratio) * output_feats

        output_feats = prepare_adapter_output(model, output_feats, targets, args.num_adapters)

        # if 'module' in dir(model):
        #     output_feats = model.module.weight_average(x, output_feats)
        # else:
        #     output_feats = model.weight_average(x, output_feats)
        
        #**********************************************
        output_feats = output_feats / output_feats.norm(dim=-1, keepdim=True)
        logits = (output_feats @ text_feature.t()) * tau
        
        labels = torch.cat([target['labels'] for target in targets])
        
        if labels.numel() == 0:
            loss_cls = logits.sum() * 0.0
        else:
            loss_cls = cross_entropy(logits, labels)
        
        loss_dict_cls = {"cls_loss": loss_cls}
            
        loss_dict = dict()
        weight_dict = dict()
        if args.use_proposal:
            class_agnostic_targets = targets.copy()
            for target in class_agnostic_targets:
                target['labels'] = target['labels'] * 0
            loss_dict = criterion(outputs, class_agnostic_targets)
            weight_dict = criterion.weight_dict
        
        loss_dict.update(loss_dict_cls)
        weight_dict.update(dict(
            cls_loss=1.0
        ))
        
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}.\n  Training terminated.".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        #************* 梯度累加 *************
        if args.grad_accumulation:
            losses = losses / accumulation_steps
            
            if (iteration+1) % accumulation_steps == 0:
                optimizer.zero_grad()
                losses.backward()
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
            else:
                losses.backward()
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            optimizer.zero_grad()
            losses.backward()
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(name)
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
        #************* 梯度累加 *************

        # optimizer.zero_grad()
        # losses.backward()
        # if max_norm > 0:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        # optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        del samples
        del targets
        del loss_dict
        del loss_dict_reduced
        del loss_dict_reduced_unscaled
        del losses
        del losses_reduced_scaled
        
        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, args=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    if args.export:
        label_map = dict()
        coco_evaluator = None
        panoptic_evaluator = None
    else:
        if args.dataset_file == 'lvis':
            from lvis import LVISEval, LVISResults
            cat2label = data_loader.dataset.cat2label
            label2cat = {v: k for k, v in cat2label.items()}
            panoptic_evaluator = None
            coco_evaluator = None
        else:
            iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
            coco_evaluator = CocoEvaluator(base_ds, iou_types, label2cat=data_loader.dataset.label2catid)

            panoptic_evaluator = None
            if 'panoptic' in postprocessors.keys():
                panoptic_evaluator = PanopticEvaluator(
                    data_loader.dataset.ann_file,
                    data_loader.dataset.ann_folder,
                    output_dir=os.path.join(output_dir, "panoptic_eval"),
                )


    print_freq = 100
    _cnt = 0
    results = []
    for samples, targets, iteration in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v if isinstance(v, (list, dict)) else v.to(device) for k, v in t.items()} for t in targets]
        
        outputs = model(samples, categories=data_loader.dataset.category_list) # samples:bs,3,w,h
        features, text_feature, tau = outputs['features'], outputs['text_feature'], outputs['tau'] # features[0]['tensors']:1,1024,50,75
        
        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict
        if args.eval_box_from == 'GT':
            if args.box_conditioned_pe:
                xywh_gt = torch.cat([target['boxes'] for target in targets])
                box_emb = gen_sineembed_for_position(xywh_gt.unsqueeze(0))[0]
                if args.only_box_size:
                    box_emb = box_emb[:,256:]
            else:
                box_emb = None

            ori_boxes = [box_cxcywh_to_xyxy(target['boxes']) for target in targets]
            box_scores = [1 for box in ori_boxes]
            num_boxes = [target['boxes'].size(0) for target in targets]
        elif args.eval_box_from == 'proposal':
            ori_boxes = [box_cxcywh_to_xyxy(box) for box in outputs['pred_boxes']]
            box_scores = [logit.sigmoid() for logit in outputs['pred_logits']]
            num_boxes = [box.size(0) for box in ori_boxes]
            
        masks = features[0].decompose()[1]
        sizes = [((1 - m[0].float()).sum(), (1 - m[:,0].float()).sum()) for m in masks]
        boxes = [box.clone() for box in ori_boxes]
        for i in range(len(boxes)):
            boxes[i][:,[0,2]] = boxes[i][:,[0,2]] * sizes[i][0]
            boxes[i][:,[1,3]] = boxes[i][:,[1,3]] * sizes[i][1]
        
        if args.roi_feat == 'layer4':
            if args.backbone == 'clip_RN50x4':
                reso = 9
            else:
                reso = 7
            roi_features = torchvision.ops.roi_align(
                features[0].tensors,
                boxes,
                output_size=(reso, reso),
                spatial_scale=1.0,
                aligned=True)  # (bs * num_queries, c, 7, 7)
        
            if 'module' in dir(model):
                output_feats = model.module.backbone[0].attnpool(roi_features, box_emb)
            else:
                output_feats = model.backbone[0].attnpool(roi_features, box_emb)
                
        elif args.roi_feat == 'layer3':
            if args.backbone == 'clip_RN50x4':
                reso = 18
            else:
                reso = 14
            roi_features = torchvision.ops.roi_align(
                features[0].tensors,
                boxes,
                output_size=(reso, reso),
                spatial_scale=1.0,
                aligned=True)  # (bs * num_queries, c, 7, 7)
        
            if 'module' in dir(model):
                output_feats = model.module.backbone[0].layer4(roi_features)
                output_feats = model.module.backbone[0].attnpool(output_feats, box_emb)
            else:
                output_feats = model.backbone[0].layer4(roi_features)
                output_feats = model.backbone[0].attnpool(output_feats, box_emb) # n(框的个数),1024

        #************* similarity
        # for target in targets:
        #     if target['image_id'].item() == 4395:
        #         labels_list = target['labels'].tolist()
        #         if 23 in labels_list:
        #             num_id = labels_list.index(23)
        #             torch.save(output_feats[num_id], 'tie_4395_embedding.pth')
        # tie_4395_embedding = torch.load("tie_4395_embedding.pth", map_location='cpu').cuda()
        # num = []
        # for output_feat_idx, output_feat in enumerate(output_feats):
        #     output_feat_norm = F.normalize(output_feat, p=2, dim=0)
        #     tie_4395_embedding_norm = F.normalize(tie_4395_embedding, p=2, dim=0)
        #     similarity = torch.dot(output_feat_norm, tie_4395_embedding_norm).item()
        #     if similarity > 0.8:
        #         num.append(output_feat_idx)
        # with open('iamge_id_label_0.8.txt', 'a') as file:
        #     for target in targets:
        #         image_id = target['image_id']
        #         for number in num:
        #             category_list = data_loader.dataset.category_list
        #             label = category_list[target['labels'][number].tolist()]
        #             file.write(f"image_id: {image_id}, label:{label}\n")
        #*************
                
        
        #********************************************** adapter
        # if 'module' in dir(model):
        #     x = model.module.shape_adapter(output_feats)
        # else:
        #     x = model.shape_adapter(output_feats)
        # ratio = 0.2
        # output_feats = ratio * x + (1 - ratio) * output_feats

        # if 'module' in dir(model):
        #     # adapter_output_xiaoyu1 = model.module.shape_adapter_xiaoyu1(output_feats)
        #     # adapter_output_dayu1 = model.module.shape_adapter_dayu1(output_feats)
        #     adapter_output_1 = model.module.shape_adapter_1(output_feats)
        #     adapter_output_2 = model.module.shape_adapter_2(output_feats)
        #     adapter_output_3 = model.module.shape_adapter_3(output_feats)
        #     adapter_output_4 = model.module.shape_adapter_4(output_feats)
        #     adapter_output_5 = model.module.shape_adapter_5(output_feats)
        #     adapter_output_6 = model.module.shape_adapter_6(output_feats)
        #     adapter_output_7 = model.module.shape_adapter_7(output_feats)
        #     adapter_output_8 = model.module.shape_adapter_8(output_feats)
        #     adapter_output_9 = model.module.shape_adapter_9(output_feats)
        #     adapter_output_10 = model.module.shape_adapter_10(output_feats)
        #     adapter_output_11 = model.module.shape_adapter_11(output_feats)
        #     adapter_output_12 = model.module.shape_adapter_12(output_feats)
        #     adapter_output_13 = model.module.shape_adapter_13(output_feats)
        #     adapter_output_14 = model.module.shape_adapter_14(output_feats)
        #     adapter_output_15 = model.module.shape_adapter_15(output_feats)
        #     adapter_output_16 = model.module.shape_adapter_16(output_feats)
        #     adapter_output_17 = model.module.shape_adapter_17(output_feats)
        #     adapter_output_18 = model.module.shape_adapter_18(output_feats)
        #     adapter_output_19 = model.module.shape_adapter_19(output_feats)
        #     adapter_output_20 = model.module.shape_adapter_20(output_feats)
        # else:
        #     # adapter_output_xiaoyu1 = model.shape_adapter_xiaoyu1(output_feats)
        #     # adapter_output_dayu1 = model.shape_adapter_dayu1(output_feats)
        #     adapter_output_1 = model.shape_adapter_1(output_feats)
        #     adapter_output_2 = model.shape_adapter_2(output_feats)
        #     adapter_output_3 = model.shape_adapter_3(output_feats)
        #     adapter_output_4 = model.shape_adapter_4(output_feats)
        #     adapter_output_5 = model.shape_adapter_5(output_feats)
        #     adapter_output_6 = model.shape_adapter_6(output_feats)
        #     adapter_output_7 = model.shape_adapter_7(output_feats)
        #     adapter_output_8 = model.shape_adapter_8(output_feats)
        #     adapter_output_9 = model.shape_adapter_9(output_feats)
        #     adapter_output_10 = model.shape_adapter_10(output_feats)
        #     adapter_output_11 = model.shape_adapter_11(output_feats)
        #     adapter_output_12 = model.shape_adapter_12(output_feats)
        #     adapter_output_13 = model.shape_adapter_13(output_feats)
        #     adapter_output_14 = model.shape_adapter_14(output_feats)
        #     adapter_output_15 = model.shape_adapter_15(output_feats)
        #     adapter_output_16 = model.shape_adapter_16(output_feats)
        #     adapter_output_17 = model.shape_adapter_17(output_feats)
        #     adapter_output_18 = model.shape_adapter_18(output_feats)
        #     adapter_output_19 = model.shape_adapter_19(output_feats)
        #     adapter_output_20 = model.shape_adapter_20(output_feats)

        # which_adapter = [] # ['1', '2', '1', '2', '2', '1', '2', '1']
        # for target in targets:
        #     for box in target['ori_boxes']:
        #         w = box[2]
        #         h = box[3]
        #         if w/h <= 1/10:
        #             which_adapter.append('1')
        #         elif 1/10< w/h <= 1/9:
        #             which_adapter.append('2')
        #         elif 1/9< w/h <= 1/8:
        #             which_adapter.append('3')
        #         elif 1/8< w/h <= 1/7:
        #             which_adapter.append('4')
        #         elif 1/7< w/h <= 1/6:
        #             which_adapter.append('5')
        #         elif 1/6< w/h <= 1/5:
        #             which_adapter.append('6')
        #         elif 1/5< w/h <= 1/4:
        #             which_adapter.append('7')
        #         elif 1/4< w/h <= 1/3:
        #             which_adapter.append('8')
        #         elif 1/3< w/h <= 1/2:
        #             which_adapter.append('9')
        #         elif 1/2< w/h <= 1/1:
        #             which_adapter.append('10')
        #         elif 1/1< w/h <= 2/1:
        #             which_adapter.append('11')
        #         elif 2/1< w/h <= 3/1:
        #             which_adapter.append('12')
        #         elif 3/1< w/h <= 4/1:
        #             which_adapter.append('13')
        #         elif 4/1< w/h <= 5/1:
        #             which_adapter.append('14')
        #         elif 5/1< w/h <= 6/1:
        #             which_adapter.append('15')
        #         elif 6/1< w/h <= 7/1:
        #             which_adapter.append('16')
        #         elif 7/1< w/h <= 8/1:
        #             which_adapter.append('17')
        #         elif 8/1< w/h <= 9/1:
        #             which_adapter.append('18')
        #         elif 9/1< w/h <= 10/1:
        #             which_adapter.append('19')
        #         elif 10/1< w/h:
        #             which_adapter.append('20')

        # adapter_outputs = []
        # for adapter_num, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7, tensor_8, tensor_9, tensor_10, \
        #     tensor_11, tensor_12, tensor_13, tensor_14, tensor_15, tensor_16, tensor_17, tensor_18, tensor_19, tensor_20 \
        #     in zip(which_adapter, adapter_output_1, adapter_output_2, adapter_output_3, adapter_output_4, adapter_output_5, \
        #            adapter_output_6, adapter_output_7, adapter_output_8, adapter_output_9, adapter_output_10, \
        #            adapter_output_11, adapter_output_12, adapter_output_13, adapter_output_14, adapter_output_15, \
        #            adapter_output_16, adapter_output_17, adapter_output_18, adapter_output_19, adapter_output_20,
        #             ):
        #     # 根据适配器选择线性层
        #     if adapter_num == '1':
        #         adapter_output = tensor_1 * 1 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 + \
        #                          tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 0 
        #         adapter_outputs.append(adapter_output)
        #     elif adapter_num == '2':
        #         adapter_output = tensor_1 * 0 + tensor_2 * 1 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 + \
        #                          tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 0
        #         adapter_outputs.append(adapter_output)
        #     elif adapter_num == '3':
        #         adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 1 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 + \
        #                          tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 0
        #         adapter_outputs.append(adapter_output)
        #     elif adapter_num == '4':
        #         adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 1 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 + \
        #                          tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 0
        #         adapter_outputs.append(adapter_output)
        #     elif adapter_num == '5':
        #         adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 1 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 + \
        #                          tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 0
        #         adapter_outputs.append(adapter_output)
        #     elif adapter_num == '6':
        #         adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 1 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 + \
        #                          tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 0
        #         adapter_outputs.append(adapter_output)
        #     elif adapter_num == '7':
        #         adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 1 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 + \
        #                          tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 0
        #         adapter_outputs.append(adapter_output)
        #     elif adapter_num == '8':
        #         adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 1 + tensor_9 * 0 + tensor_10 * 0 + \
        #                          tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 0
        #         adapter_outputs.append(adapter_output)
        #     elif adapter_num == '9':
        #         adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 1 + tensor_10 * 0 + \
        #                          tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 0
        #         adapter_outputs.append(adapter_output)
        #     elif adapter_num == '10':
        #         adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 1 + \
        #                          tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 0
        #         adapter_outputs.append(adapter_output)
        #     elif adapter_num == '11':
        #         adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 + \
        #                          tensor_11 * 1 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 0 
        #         adapter_outputs.append(adapter_output)
        #     elif adapter_num == '12':
        #         adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 + \
        #                          tensor_11 * 0 + tensor_12 * 1 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 0
        #         adapter_outputs.append(adapter_output)
        #     elif adapter_num == '13':
        #         adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 + \
        #                          tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 1 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 0
        #         adapter_outputs.append(adapter_output)
        #     elif adapter_num == '14':
        #         adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 + \
        #                          tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 1 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 0
        #         adapter_outputs.append(adapter_output)
        #     elif adapter_num == '15':
        #         adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 + \
        #                          tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 1 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 0
        #         adapter_outputs.append(adapter_output)
        #     elif adapter_num == '16':
        #         adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 + \
        #                          tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 1 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 0
        #         adapter_outputs.append(adapter_output)
        #     elif adapter_num == '17':
        #         adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 + \
        #                          tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 1 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 0
        #         adapter_outputs.append(adapter_output)
        #     elif adapter_num == '18':
        #         adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 + \
        #                          tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 1 + tensor_19 * 0 + tensor_20 * 0
        #         adapter_outputs.append(adapter_output)
        #     elif adapter_num == '19':
        #         adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 + \
        #                          tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 1 + tensor_20 * 0
        #         adapter_outputs.append(adapter_output)
        #     elif adapter_num == '20':
        #         adapter_output = tensor_1 * 0 + tensor_2 * 0 + tensor_3 * 0 + tensor_4 * 0 + tensor_5 * 0 + tensor_6 * 0 + tensor_7 * 0 + tensor_8 * 0 + tensor_9 * 0 + tensor_10 * 0 + \
        #                          tensor_11 * 0 + tensor_12 * 0 + tensor_13 * 0 + tensor_14 * 0 + tensor_15 * 0 + tensor_16 * 0 + tensor_17 * 0 + tensor_18 * 0 + tensor_19 * 0 + tensor_20 * 1
        #         adapter_outputs.append(adapter_output)
        #     else:
        #         raise ValueError("Invalid adapter value")
        # x = torch.stack(adapter_outputs, dim=0)

        # ratio = 0.2
        # output_feats = ratio * x + (1 - ratio) * output_feats

        output_feats = prepare_adapter_output(model, output_feats, targets, args.num_adapters)

        # if 'module' in dir(model):
        #     output_feats = model.module.weight_average(x, output_feats)
        # else:
        #     output_feats = model.weight_average(x, output_feats)
        #**********************************************

        output_feats = output_feats / output_feats.norm(dim=-1, keepdim=True)
        logits = (output_feats @ text_feature.t()) * tau
        
        labels = torch.cat([target['labels'] for target in targets])
        if args.export:
            pred_labels = logits.argmax(dim=-1)
            box_ids = torch.cat([target['box_ids'] for target in targets])
            for id, label in zip(box_ids, pred_labels):
                label_map[id.item()] = data_loader.dataset.label2catid[label.item()]
        
        # loss = cross_entropy(logits, labels)
        
        # loss_dict = {"ce_loss": loss}

        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = loss_dict_reduced
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # results = postprocessors['bbox'](outputs, orig_target_sizes)
        
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = orig_target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        if args.dataset_file == 'coco':
            results = []
        logits = logits.softmax(dim=-1)
        if args.dataset_file == 'lvis':
            for logit, box, scale, box_score, target in zip(logits.split(num_boxes), ori_boxes, scale_fct, box_scores, targets):
                logit = logit * box_score
                scores, indices = logit.flatten().topk(k=min(300, logit.numel()))
                box_id = torch.div(indices, logit.size(1), rounding_mode='floor')
                cls_id = indices % logit.size(1)
                pred_boxes = box[box_id]
                image_id = target['image_id'].item()
                out_boxes = pred_boxes * scale[None]
                out_boxes = convert_to_xywh(out_boxes)
                
                for ind in range(len(scores)):
                    temp = {
                        "image_id": image_id,
                        "score": scores[ind].item(),
                        "category_id": cls_id[ind].item(),
                        "bbox": out_boxes[ind].tolist(),
                    }
                    if args.label_map:
                        temp["category_id"] = label2cat[temp["category_id"]]

                    results.append(temp)
        else:
            for logit, box, scale, box_score in zip(logits.split(num_boxes), ori_boxes, scale_fct, box_scores):
                logit = logit * box_score
                scores, indices = logit.flatten().topk(k=min(100, logit.numel()))
                box_id = torch.div(indices, logit.size(1), rounding_mode='floor')
                cls_id = indices % logit.size(1)
                pred_boxes = box[box_id]
                results.append(dict(
                    scores=scores,
                    labels=cls_id,
                    boxes=pred_boxes * scale[None],
                ))
        
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name
            panoptic_evaluator.update(res_pano)
            
        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if args.export:
        import json
        with open(f'logs/export_label_{utils.get_rank()}.json', 'w') as f:
            json.dump(label_map, f)
            

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    
    if args.dataset_file == 'lvis':
        rank = utils.get_rank()
        torch.save(results, output_dir + f"/pred_{rank}.pth")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if rank == 0:
            world_size = utils.get_world_size()
            for i in range(1, world_size):
                temp = torch.load(output_dir + f"/pred_{i}.pth")
                results += temp


        lvis_results = LVISResults(base_ds, results, max_dets=300)
        lvis_eval = LVISEval(base_ds, lvis_results, "bbox")
        lvis_eval.run()
        lvis_eval.print_results()
    
    del samples
    del targets
    # del loss_dict
    # del loss_dict_reduced
    # del loss_dict_reduced_unscaled

    torch.cuda.empty_cache()

    return stats, coco_evaluator
    # return 0


