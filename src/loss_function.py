# -*- coding: UTF-8 -*-
'''

Describle:         损失函数

Created on          2024/07/24 15:24:43
Author:             @ Mr_Robot
Current State:      增加Diceloss，测试通过

'''

#! 医学图像计算loss和指标都可以不用考虑背景，而是求子区域的平均loss和指标

from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
import torch

# class GMDiceLoss:
#     def __init__(self, smooth=1e-5):
#         """
#         初始化函数:
#         :param smooth: 平滑因子
#         :param w1: ET权重
#         :param w2: TC权重
#         :param w3: WT权重
#         """
#         self.smooth = smooth
#         self.sub_areas = ['ET', 'TC', 'WT']
#         self.labels = {
#             'BG': 0, 
#             'NCR' : 1,
#             'ED': 2,
#             'ET':3
#         }
#         self.num_classes = len(self.labels)

#     def __call__(self, y_pred, y_mask):
#         """
#         DiceLoss
#         :param y_pred: 预测值 [batch, 4, D, W, H]
#         :param y_mask: 真实值 [batch, D, W, H]
#         """
#         tensor_one = torch.tensor(1)
#         y_mask = F.one_hot(y_mask, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float() # y_mask ==> [batch, 4, 144, 128, 128]
#         area_et_loss, area_tc_loss, area_wt_loss = self.get_every_subAreas_loss(y_pred, y_mask)

#         mean_loss = (area_et_loss + area_tc_loss + area_wt_loss) / 3
#         return mean_loss, area_et_loss, area_tc_loss, area_wt_loss

#     def get_every_subAreas_loss(self, y_pred, y_mask):
#         loss_dict = {}
#         pred_list, mask_list = splitSubAreas(y_pred, y_mask)

#         # 计算子区域的diceloss
#         for sub_area, pred, mask in zip(self.sub_areas, pred_list, mask_list):
#             intersection = (pred * mask).sum(dim=(-3, -2, -1))
#             union = pred.sum(dim=(-3, -2, -1)) + mask.sum(dim=(-3, -2, -1))
#             dice_c = 2 * (intersection + self.smooth) / (union + self.smooth)
#             loss_dict[sub_area] = 1 - dice_c.mean()

#         # 计算batch平均损失
#         area_et_loss = loss_dict['ET']
#         area_tc_loss = loss_dict['TC']
#         area_wt_loss = loss_dict['WT']

#         return area_et_loss, area_tc_loss, area_wt_loss

class DiceLoss:
    def __init__(self, smooth=1e-5, w1=0.2, w2=0.2, w3=0.4):
        """
        初始化函数:
        :param smooth: 平滑因子
        :param w1: ET权重
        :param w2: TC权重
        :param w3: WT权重
        """
        self.smooth = smooth
        self.sub_areas = ['ET', 'TC', 'WT']
        self.labels = {
            'BG': 0, 
            'NCR' : 1,
            'ED': 2,
            'ET':3
        }
        self.num_classes = len(self.labels)

    def __call__(self, y_pred, y_mask):
        """
        DiceLoss
        :param y_pred: 预测值 [batch, 4, D, W, H]
        :param y_mask: 真实值 [batch, D, W, H]
        """
        y_mask = F.one_hot(y_mask, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float() # y_mask ==> [batch, 4, 144, 128, 128]
        area_et_loss, area_tc_loss, area_wt_loss = self.get_every_subAreas_loss(y_pred, y_mask)

        # intersection = (y_pred * y_mask).sum(dim=(-3, -2, -1))
        # union = y_pred.sum(dim=(-3, -2, -1)) + y_mask.sum(dim=(-3, -2, -1))
        # dice = (2. * intersection + self.smooth) / (union + self.smooth)
        # loss = 1. - dice.mean()  
        
        mean_loss = (area_et_loss + area_tc_loss +  area_wt_loss) / 3
        return mean_loss, area_et_loss, area_tc_loss, area_wt_loss

    def get_every_subAreas_loss(self, y_pred, y_mask):
        loss_dict = {}
        pred_list, mask_list = splitSubAreas(y_pred, y_mask)

        # 计算子区域的diceloss
        for sub_area, pred, mask in zip(self.sub_areas, pred_list, mask_list):
            intersection = (pred * mask).sum(dim=(-3, -2, -1))
            union = pred.sum(dim=(-3, -2, -1)) + mask.sum(dim=(-3, -2, -1))
            dice_c = (2. * intersection + self.smooth) / (union + self.smooth)
            # assert (dice_c > 0).all() and (dice_c < 1).any(), "DiceLoss Error: dice_c mast be between 0 and 1"
            loss_dict[sub_area] = 1. - dice_c.mean()

        # 计算batch平均损失
        area_et_loss = loss_dict['ET']
        area_tc_loss = loss_dict['TC']
        area_wt_loss = loss_dict['WT']

        # assert not ((area_et_loss < 0).any() or (area_tc_loss < 0).any() or (area_wt_loss < 0).any()), f"DiceLoss Error: loss < 0, {area_et_loss}, {area_tc_loss}, {area_wt_loss}"
        return area_et_loss, area_tc_loss, area_wt_loss

# Focal Loss
class FocalLoss:
    def __init__(self, loss_type='mean', gamma=2, alpha=0.8, w1=0.3, w2=0.3, w3=0.3, w_bg=0.1):
        """
        初始化函数:
        :param loss_type: loss计算方式，可选['custom', 'mean']，默认为'custom'
        :param gamma: 
        :param alpha: 
        :param w1: ET权重
        :param w2: TC权重
        :param w3: WT权重
        """
        self.loss_type = loss_type
        self.gamma = gamma
        self.alpha = alpha
        self.sub_areas = ['ET', 'TC', 'WT']
        self.labels = {
            'BG': 0, 
            'NCR' : 1,
            'ED': 2,
            'ET':3
        }
        self.num_classes = len(self.labels)
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w_bg = w_bg
        
    def __call__(self, y_pred, y_mask):
        """
        FocalLoss
        :param y_pred: 预测值 [batch, 4, D, W, H]
        :param y_mask: 真实值 [batch, D, W, H]
        
        """
        loss_dict = {}
        y_mask = F.one_hot(y_mask, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()

        pred_list, mask_list = splitSubAreas(y_pred, y_mask)
        for sub_area, pred, mask in zip(self.sub_areas, pred_list, mask_list):
            loss = self.cal_focal_loss(pred, mask)
            loss_dict[sub_area] = loss 
        et_loss = loss_dict['ET']
        tc_loss = loss_dict['TC']
        wt_loss = loss_dict['WT']

        mean_loss = (et_loss + tc_loss + wt_loss) / 3

        return mean_loss, et_loss, tc_loss, wt_loss
    
    def cal_focal_loss(self, y_pred, y_mask):
        cross_entropy = F.cross_entropy(y_pred, y_mask, reduction="mean")
        pt = torch.exp(-cross_entropy)
        focal_loss = (self.alpha * ((1 - pt) ** self.gamma) * cross_entropy)
        return focal_loss
    
# CELoss
class CELoss:
    def __init__(self, loss_type='mean', smooth=1e-5, w1=0.3, w2=0.3, w3=0.3, w_bg=0.1):
        """
        初始化函数:
        :param smooth: 平滑因子
        :param w1: ET权重
        :param w2: TC权重
        :param w3: WT权重
        """
        self.smooth = smooth
        self.loss_type = loss_type
        self.sub_areas = ['ET', 'TC', 'WT'] # 异常子区域只有后三个
        self.labels = {
            'BG': 0,  # 背景
            'NCR' : 1, # 
            'ED': 2, # 增强肿瘤
            'ET':3
        }
        self.num_classes = len(self.labels)

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w_bg = w_bg

    def __call__(self, y_pred, y_mask):
        """
        CrossEntropyLoss
        :param y_pred: 预测值 [batch, 4, D, W, H]
        :param y_mask: 真实值 [batch, D, W, H]
        """
        loss_dict = {}
        
        y_mask = F.one_hot(y_mask, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        pred_list, mask_list = splitSubAreas(y_pred, y_mask)

        for sub_area, pred, mask in zip(self.sub_areas, pred_list, mask_list):
            CEloss = F.cross_entropy(pred, mask, reduction="mean")
            loss_dict[sub_area] = CEloss

        et_loss = loss_dict['ET']
        tc_loss = loss_dict['TC']
        wt_loss = loss_dict['WT']
        
        mean_loss = (et_loss + tc_loss + wt_loss) / 3
        
        return mean_loss, et_loss, tc_loss, wt_loss        


def splitSubAreas(y_pred, y_mask):
    """
    分割出子区域
    :param y_pred: 预测值 [batch, 4, D, W, H]
    :param y_mask: 真实值 [batch, 4, D, W, H]
    """
    et_pred = y_pred[:, 3,...]
    tc_pred = y_pred[:, 1,...] + y_pred[:,3,...]
    wt_pred = y_pred[:, 1:,...].sum(dim=1)
    
    et_mask = y_mask[:, 3,...]
    tc_mask = y_mask[:, 1,...] + y_mask[:,3,...]
    wt_mask = y_mask[:, 1:,...].sum(dim=1)
    
    pred_list = [et_pred, tc_pred, wt_pred]
    mask_list = [et_mask, tc_mask, wt_mask]
    return pred_list, mask_list


if __name__ == '__main__':
    y_pred = torch.randn(2, 4, 128, 128, 128)
    # y_pred = torch.argmax(y_pred, dim=1)
    y_mask = torch.randint(0, 4, (2, 128, 128, 128))
    diceLossFunc = DiceLoss()
    ceLossFunc = CELoss()
    focallossFunc = FocalLoss()

    print(f"CELoss : {diceLossFunc(y_pred, y_mask)}")
    
    print(f"DiceLoss : {ceLossFunc(y_pred, y_mask)}")
    
    print(f"FocalLoss : {focallossFunc(y_pred, y_mask)}")
    