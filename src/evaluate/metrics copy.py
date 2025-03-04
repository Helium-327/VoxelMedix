
# -*- coding: UTF-8 -*-
'''
Describle:         各项评估指标构建

Created on          2024/07/24 11:49:44
Author:             @ Mr_Robot
Current State:      整个流程构建完成,测试完成
Notice:             计算Dice系数时，先计算子区大小，再计算Dice系数
'''

import torch
import os
import numpy as np
from torch.nn import functional as F
from tabulate import tabulate

class EvaluationMetrics:
    def __init__(self, smooth=1e-5, num_classes=4):
        self.smooth = smooth
        self.num_classes = num_classes
        self.sub_areas = ['ET', 'TC', 'WT']

    def pre_processing(self, y_pred, y_mask):
        """
        预处理：
            1.挑选出预测概率最大的类别；
            2.one-hot处理
        :param y_pred: 预测标签
        :param y_mask: 真实标签
        :return: 处理后的预测标签和真实标签
        """
        et_pred = y_pred[:, 3, ...]
        tc_pred = y_pred[:, 1, ...] + y_pred[:, 3, ...]
        wt_pred = y_pred[:, 1:, ...].sum(dim=1)
         
        et_mask = y_mask[:, 3, ...]
        tc_mask = y_mask[:, 1, ...] + y_mask[:, 3, ...]
        wt_mask = y_mask[:, 1:, ...].sum(dim=1)
        
        pred_list = [et_pred, tc_pred, wt_pred]
        mask_list = [et_mask, tc_mask, wt_mask]
        
        return pred_list, mask_list
    
    def calculate_confusion_matrix(self, y_pred, y_mask):
        """
        计算混淆矩阵元素值
        :param y_pred: 预测标签
        :param y_mask: 真实标签
        :return: 混淆矩阵元素值TP、FN、 FP、TN
        """
        assert y_pred.shape == y_mask.shape, "预测标签和真实标签的维度必须相同"
        tensor_one = torch.tensor(1)
        
        # 计算混淆矩阵的元素,在类别维度取平均
        TP = (y_pred * y_mask).sum(dim=(-3, -2, -1)) # 预测为正类，实际也为正类
        FN = ((tensor_one - y_pred) * y_mask).sum(dim=(-3, -2, -1)) # 预测为负类，实际为正类
        FP = (y_pred * (tensor_one - y_mask)).sum(dim=(-3, -2, -1)) # 预测为正类，实际为负类
        TN = ((tensor_one - y_pred) * (tensor_one - y_mask)).sum(dim=(-3, -2, -1)) # 预测为负类，实际也为负类
        
        return TP, FN, FP, TN
    
    def dice_coefficient(self, y_pred, y_mask):
        """
        计算Dice 系数
        :param y_pred: 预测标签
        :param y_mask: 真实标签
        :return: Dice 系数
        """
        dice_coeffs = {}
        # 预处理
        y_pred = torch.argmax(y_pred, dim=1).to(dtype=torch.int64) # 降维，选出概率最大的类索引值
        y_pred = F.one_hot(y_pred, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float() # one-hot
        y_mask = F.one_hot(y_mask, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float() # one-hot

        # 计算每个类别的Dice系数
        pred_list, mask_list = self.pre_processing(y_pred, y_mask) # 获取子区的预测标签和真实标签

        for sub_area, sub_pred, sub_mask in zip(self.sub_areas, pred_list, mask_list):
            intersection = (sub_pred * sub_mask).sum(dim=(-3, -2, -1))
            union = sub_pred.sum(dim=(-3, -2, -1)) + sub_mask.sum(dim=(-3, -2, -1))
            dice_c = (2. * intersection ) / (union + self.smooth)
            dice_coeffs[sub_area] = dice_c.mean()
        
        intersection = (y_pred * y_mask).sum(dim=(-3, -2, -1))
        union = y_pred.sum(dim=(-3, -2, -1)) + y_mask.sum(dim=(-3, -2, -1))
        dice = (2. * intersection ) / (union + self.smooth)
        dice_coeffs['global mean'] = dice.mean()

        # 提取特定类别的Dice系数
        et_dice = dice_coeffs['ET'].item()
        tc_dice = dice_coeffs['TC'].item()
        wt_dice = dice_coeffs['WT'].item()
        # global_mean_dice = dice_coeffs['global mean'].item()

        mean_dice = (et_dice + tc_dice + wt_dice) / 3
        
        return mean_dice, et_dice, tc_dice, wt_dice
    
    def jaccard_index(self, y_pred, y_mask):
        """
        计算Jaccard 系数
        :param y_pred: 预测标签
        :param y_mask: 真实标签
        :return: Jaccard 系数 (全局平均)
        """
        # 获取子区的预测标签和真实标签
        y_pred = torch.argmax(y_pred, dim=1).to(dtype=torch.int64) # 降维，选出概率最大的类索引值
        y_pred = F.one_hot(y_pred, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float() # one-hot
        y_mask = F.one_hot(y_mask, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float() # one-hot
        
        pred_list, mask_list = self.pre_processing(y_pred, y_mask) 
        jaccard_coeffs = {}

        # 计算每个类别的Jaccard系数
        for sub_area, pred, mask in zip(self.sub_areas, pred_list, mask_list):
            intersection = (pred * mask).sum(dim=(-3, -2, -1))
            union = pred.sum(dim=(-3, -2, -1)) + mask.sum(dim=(-3, -2, -1)) - intersection
            jaccard = (intersection)/ (union + self.smooth)
            jaccard_coeffs[sub_area] = jaccard.mean()

        # intersection = (y_pred * y_mask).sum(dim=(-3, -2, -1))
        # union = y_pred.sum(dim=(-3, -2, -1)) + y_mask.sum(dim=(-3, -2, -1)) - intersection
        # jaccard = (intersection) / (union + self.smooth)
        # jaccard_coeffs['global mean'] = jaccard.mean()
        # global_mean_dice = jaccard_coeffs['global mean'].item()
        
        # 提取特定类别的Jaccard系数
        et_jaccard = jaccard_coeffs['ET'].item()
        tc_jaccard = jaccard_coeffs['TC'].item()
        wt_jaccard = jaccard_coeffs['WT'].item()
        
        mean_jaccard = (et_jaccard + tc_jaccard + wt_jaccard) / 3
        
        return mean_jaccard, et_jaccard, tc_jaccard, wt_jaccard
    
    def recall(self, y_pred, y_mask):
        """
        计算Recall(查全率，敏感性（真阳性率））
        :param y_pred: 预测标签
        :param y_mask: 真实标签
        :return: 敏感性/特异性（全局平均 、ET 、TC、WT）
        """
        recall_scores = {}
        y_pred = torch.argmax(y_pred, dim=1).to(dtype=torch.int64) # 降维，选出概率最大的类索引值
        y_pred = F.one_hot(y_pred, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float() # one-hot
        y_mask = F.one_hot(y_mask, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float() # one-hot
        
        # 获取子区的预测标签和真实标签
        pred_list, mask_list = self.pre_processing(y_pred, y_mask) 

        # 计算子区域的混淆矩阵的元素
        for sub_area, pred, mask in zip(self.sub_areas, pred_list, mask_list):
            TP, FN, _, _ = self.calculate_confusion_matrix(pred, mask)
            recall = (TP) / (TP + FN + self.smooth)
            recall_scores[sub_area] = recall.mean()
        
        # # 计算全局平均Recall
        # TP, FN, _, _ = self.calculate_confusion_matrix(y_pred, y_mask)
        # recall = (TP) / (TP + FN + self.smooth)
        # recall_scores['global_mean'] = recall.mean()
        # global_mean_recall = recall_scores['global_mean'].item()
        
        
        et_recall = recall_scores['ET'].item()
        tc_recall = recall_scores['TC'].item()
        wt_recall = recall_scores['WT'].item()

        mean_recall = (et_recall + tc_recall + wt_recall) / 3

        return mean_recall, et_recall, tc_recall, wt_recall 
    
    def precision(self, y_pred, y_mask):
        """
        计算Precision （查准率，特异性（真阴性率））
        :param y_pred: 预测标签
        :param y_mask: 真实标签
        :return: 全局平均查准率， ET查准率，TC查准率，WT查准率
        """
        precision_scores = {}
        
        y_pred = torch.argmax(y_pred, dim=1).to(dtype=torch.int64) # 降维，选出概率最大的类索引值
        y_pred = F.one_hot(y_pred, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float() # one-hot
        y_mask = F.one_hot(y_mask, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float() # one-hot
        
        # 获取子区的预测标签和真实标签
        pred_list, mask_list = self.pre_processing(y_pred, y_mask) 

        # 计算混淆矩阵的元素
        for sub_area, pred, mask in zip(self.sub_areas, pred_list, mask_list):
            TP, _, FP, _ = self.calculate_confusion_matrix(pred, mask)
            precision = (TP ) / (TP + FP + self.smooth)
            precision_scores[sub_area] = precision.mean()
        
        
        # TP, _, FP, _ = self.calculate_confusion_matrix(y_pred, y_mask)
        # precision = (TP ) / (TP + FP + self.smooth)
        # precision_scores['global_mean'] = precision.mean()
        # global_mean_precision = precision_scores['global_mean'].item()
        
        et_precision = precision_scores['ET'].item()
        tc_precision = precision_scores['TC'].item()
        wt_precision = precision_scores['WT'].item()

        mean_precision = (et_precision + tc_precision + wt_precision) / 3
        
        return mean_precision, et_precision, tc_precision, wt_precision
        
    def accuracy(self, y_pred, y_mask):
        """
        准确率
        :param y_pred: 预测标签
        :param y_mask: 真实标签
        :return: 准确率
        """
        accuracy_scores = {}
        y_pred = torch.argmax(y_pred, dim=1).to(dtype=torch.int64) # 降维，选出概率最大的类索引值
        y_pred = F.one_hot(y_pred, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float() # one-hot
        y_mask = F.one_hot(y_mask, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float() # one-hot

        # 获取子区的预测标签和真实标签
        pred_list, mask_list = self.pre_processing(y_pred, y_mask)
        for sub_area, pred, mask in zip(self.sub_areas, pred_list, mask_list):
            TP, FN, FP, TN = self.calculate_confusion_matrix(pred, mask)
            accuracy = (TP + TN) / (TP + FN + FP + TN + self.smooth)
            accuracy_scores[sub_area] = accuracy.mean()
            
        # TP, FN, FP, TN = self.calculate_confusion_matrix(y_pred, y_mask)
        # accuracy = (TP + TN) / (TP + FN + FP + TN)
        # accuracy_scores['global_mean'] = accuracy.mean()
        # global_mean_accuracy = accuracy_scores['global_mean'].item()
            
        et_accuracy = accuracy_scores['ET'].item()
        tc_accuracy = accuracy_scores['TC'].item()
        wt_accuracy = accuracy_scores['WT'].item()
        
        mean_accuracy = (et_accuracy + tc_accuracy + wt_accuracy) / 3
        
        return mean_accuracy, et_accuracy, tc_accuracy, wt_accuracy
    
    def f1_score(self, y_pred, y_mask):
        """
        计算F1值
        :param y_pred: 预测标签
        :param y_mask: 真实标签
        :return: F1值
        """
        f1_scores = {}
        precision_list = self.precision(y_pred, y_mask)
        recall_list = self.recall(y_pred, y_mask)
        
        # f1_score on ET
        f1_scores[self.sub_areas[0]] = 2 * (precision_list[1] * recall_list[1]) / (precision_list[1] + recall_list[1] + self.smooth)
        # f1_socre on TC
        f1_scores[self.sub_areas[1]] = 2 * (precision_list[2] * recall_list[2]) / (precision_list[2] + recall_list[2] + self.smooth)
        # f1_score on WT
        f1_scores[self.sub_areas[2]] = 2 * (precision_list[3] * recall_list[3]) / (precision_list[3] + recall_list[3] + self.smooth)
        
        # # f1_score on global
        # f1_scores['global_mean'] = 2 * (precision_list[0] * recall_list[0]) / (precision_list[0] + recall_list[0] + self.smooth)
        # mean_f1 = f1_scores['global_mean']
        
        et_f1 = f1_scores['ET']
        tc_f1 = f1_scores['TC']
        wt_f1 = f1_scores['WT']
        mean_f1 = (et_f1 + tc_f1 + wt_f1) / 3       
        
        return mean_f1, et_f1, tc_f1, wt_f1
    
    def f2_score(self, y_pred, y_mask):
        """
        计算F2值
        :param y_pred: 预测标签
        :param y_mask: 真实标签
        :return: F2值
        """
        f2_scores = {}
        precision_list = self.precision(y_pred, y_mask)
        recall_list = self.recall(y_pred, y_mask)
        
        # f2_score on ET
        f2_scores[self.sub_areas[0]] = (5 * precision_list[1] * recall_list[1]) / (4 * precision_list[1] + recall_list[1] + self.smooth)
        # f2_socre on TC
        f2_scores[self.sub_areas[1]] = (5 * precision_list[2] * recall_list[2]) / (4 * precision_list[2] + recall_list[2] + self.smooth)
        # f2_score on WT
        f2_scores[self.sub_areas[2]] = (5 * precision_list[3] * recall_list[3]) / (4 * precision_list[3] + recall_list[3] + self.smooth)
        
        # f2_score on global
        # f2_scores['global_mean'] = (5 * precision_list[0] * recall_list[0]) / (4 * precision_list[0] + recall_list[0] + self.smooth)
        # mean_f2 = f2_scores['global_mean']
        
        et_f2 = f2_scores['ET']
        tc_f2 = f2_scores['TC']
        wt_f2 = f2_scores['WT']
        mean_f2 = (et_f2 + tc_f2 + wt_f2) / 3

        return mean_f2, et_f2, tc_f2, wt_f2
    
    def hausdorff_95_torch(self, y_pred, y_mask): #FIXME: 显存占用太高，无法计算
        """
        使用 PyTorch 计算 Hausdorff 95 距离
        :param y_pred: 预测标签 (经过 argmax 和 one-hot 处理后的张量)
        :param y_mask: 真实标签 (经过 one-hot 处理后的张量)
        :return: Hausdorff 95 距离 (全局平均、ET、TC、WT)
        """
        h95_scores = {}
        
        # 获取子区的预测标签和真实标签
        pred_list, mask_list = self.pre_processing(y_pred, y_mask)
        
        # 计算每个子区域的 Hausdorff 95 距离
        for sub_area, pred, mask in zip(self.sub_areas, pred_list, mask_list):
            #? 1. 获取预测和真实标签中的点坐标
            pred_points = torch.nonzero(pred > 0.5, as_tuple=False).float()  # [N, 3]
            mask_points = torch.nonzero(mask > 0.5, as_tuple=False).float()  # [M, 3]
            
            if len(pred_points) == 0 or len(mask_points) == 0:
                h95_scores[sub_area] = 0.0  # 如果预测或真实标签中没有点，距离为0
                continue
            
            #? 2. 计算点集之间的距离矩阵
            dist_matrix = torch.cdist(pred_points, mask_points)  # [N, M]
            
            #? 3. 计算 Hausdorff 距离
            hd_pred_to_mask = dist_matrix.min(dim=1).values  # [N]
            hd_mask_to_pred = dist_matrix.min(dim=0).values  # [M]
            
            #? 4. 计算 Hausdorff 95 距离
            k_pred = int(0.95 * len(hd_pred_to_mask))
            k_mask = int(0.95 * len(hd_mask_to_pred))
            
            hd_pred_to_mask = torch.topk(hd_pred_to_mask, k_pred, largest=False).values[-1]
            hd_mask_to_pred = torch.topk(hd_mask_to_pred, k_mask, largest=False).values[-1]
            
            #? 5. 计算全局Hausdorff 95 距离
            h95 = max(hd_pred_to_mask, hd_mask_to_pred)
            h95_scores[sub_area] = h95.item()
        
        # 计算全局的 Hausdorff 95 距离
        pred_points = torch.nonzero(y_pred > 0.5, as_tuple=False).float()  # [N, 3]
        mask_points = torch.nonzero(y_mask > 0.5, as_tuple=False).float()  # [M, 3]
        
        if len(pred_points) == 0 or len(mask_points) == 0:
            h95_scores['global_mean'] = 0.0  # 如果预测或真实标签中没有点，距离为0
        else:
            # 计算点集之间的距离矩阵
            dist_matrix = torch.cdist(pred_points, mask_points)  # [N, M]
            
            # 计算 Hausdorff 距离
            hd_pred_to_mask = dist_matrix.min(dim=1).values  # [N]
            hd_mask_to_pred = dist_matrix.min(dim=0).values  # [M]
            
            # 取前95%的点
            k_pred = int(0.95 * len(hd_pred_to_mask))
            k_mask = int(0.95 * len(hd_mask_to_pred))
            
            hd_pred_to_mask = torch.topk(hd_pred_to_mask, k_pred, largest=False).values[-1]
            hd_mask_to_pred = torch.topk(hd_mask_to_pred, k_mask, largest=False).values[-1]
            
            # Hausdorff 95 距离
            h95 = max(hd_pred_to_mask, hd_mask_to_pred)
            h95_scores['global_mean'] = h95.item()
        
        # 提取特定类别的 Hausdorff 95 距离
        et_h95 = h95_scores['ET']
        tc_h95 = h95_scores['TC']
        wt_h95 = h95_scores['WT']
        global_mean_h95 = h95_scores['global_mean']
        
        return global_mean_h95, et_h95, tc_h95, wt_h95    
    
    def update(self, y_pred, y_mask):
        """
        更新评估指标
        :param y_pred: 预测标签
        :param y_mask: 真实标签
        :return: 所有的评估指标
        """
        dice_scores = self.dice_coefficient(y_pred, y_mask)
        jacc_scores = self.jaccard_index(y_pred, y_mask)
        accuracy_scores = self.accuracy(y_pred, y_mask)
        precision_scores = self.precision(y_pred, y_mask)
        recall_scores = self.recall(y_pred, y_mask)
        f1_scores = self.f1_score(y_pred, y_mask)
        f2_scores = self.f2_score(y_pred, y_mask)

        metrics = [dice_scores, jacc_scores, accuracy_scores, precision_scores, recall_scores, f1_scores, f2_scores]
        metrics = np.stack(metrics, axis=0) # [7, 4]
        metrics = np.nan_to_num(metrics)
        return metrics

def format_value(value, decimals=4):
    # 返回一个格式化后的字符串，保留指定的小数位数
    return f"{value:.{decimals}f}"

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_root = "./data_brats"
    data_dir = os.path.join(local_root, "BraTS2021_00621")

    data_size = (144, 224, 224)
    
    # transform = RandomCrop3D(target_size)
    brats = BraTS21_3d(data_dir, data_size=data_size)

    data, label = brats.load_image(data_dir)
    data = data[None,...].to(device)
    label = label[None,...].to(device)
    model = UNet3D(in_channels=4, num_classes=4)
    # print(model)
    model.to(device)

    y_pred = model(data)
    y_mask = label
    metrics_list = np.zeros((4, 7))
    metrics = EvaluationMetrics()
    metrics_list = metrics.update(y_pred, y_mask, metrics_list)
    # metrics.printAll(y_pred, y_mask, metrics_list)
    print(metrics_list)
        
        

            