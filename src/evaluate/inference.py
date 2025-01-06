# -*- coding: UTF-8 -*-
'''

Describle:         模型评估：在训练完成之后使用测试集对模型性能进行评估，并保存评估结果

Created on         2024/08/18 14:07:26
Author:            @ Mr_Robot
Current State:     #TODO:
'''
import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
import nibabel as nib
from matplotlib import pyplot as plt
from tabulate import tabulate

from torch.nn import functional as F
from torch.optim import RMSprop, AdamW
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from readDatasets.BraTS import BraTS21_3d
from transforms import data_transform, Compose, RandomCrop3D, Normalize, tioRandomFlip3d
# from loss_function import DiceLoss, CELoss
from metrics import *

from utils.log_writer import custom_logger
from utils.ckpt_save_load import load_checkpoint
# from utils.splitDataList import DataSpliter
from utils.get_commits import *

from nets.unet3d.fusion_unet import *
# from nets.pasnet.pspnet import PSPNET
from torchcrf import CRF
# from nets.unet3d.src.unet3d_bn import UNet3D_BN, UNet3D_ResBN, UNet3D_BN_SE, UNet3D_ResBN_SE
# from nets.unet3d.unet3d_ln import UNet3D_LN
# from nets.unet3d.unet3d_CBAM import unet3d_CBAM
# from nets.unet3d.unet3d_5x5 import UNet3D_BN_5x5
# from nets.unet3d.uent3d_dilation import UNet3D_dilation, UNet3D_ResDilation
# from nets.unet3d.temp.uent3d_ResSE import UNet3D_ResSE
# from nets.unet3d.pspnet import PSPNET

from utils.plot_tools.plot_results import NiiViewer

date_str = get_current_date() + '_' + get_current_time()


def inference(test_loader, model, Metricer, output_path, device, affine, window_size=(128, 128, 128), stride_ratio=0.5, save_flag=False):
    """
    验证过程：
        1. 使用滑窗预测算法对测试集数据进行推理预测
        2. 保存预测结果nii文件
        3. 使用评估指标对预测结果进行评估
        4. 保存评估结果
    :param model: 模型
    :param metrics: 评估指标
    :param train_loader: 训练数据集
    :param test_loader: 验证数据集
    :param scaler: 缩放器
    :param optimizer: 优化器
    :param loss_funtion: 损失函数
    :param device: 设备
    :param model_path: 模型路径
    """
    Metrics_list = np.zeros((7, 4))
    for i, data in enumerate(tqdm(test_loader)):
        vimage, vmask = data[0], data[1]
        # crf = CRF(data[0].shape[1], batch_first=True)
        pred_vimage = slide_window_pred(model, vimage, device, window_size=window_size, stride_ratio=stride_ratio)

        # 保存预测结果nii文件
        if save_flag:
            save_nii(pred_vimage, vimage, vmask, output_path, affine, i)

        # 评估指标
        metrics = Metricer.update(pred_vimage, vmask)
        Metrics_list += metrics
    Metrics_list /= len(test_loader)

    test_scorce = {}
    # 记录验证结果
    test_scorce['Dice_scores'] = Metrics_list[0] 
    test_scorce['Jaccard_scores'] = Metrics_list[1]
    test_scorce['Accuracy_scores'] = Metrics_list[2]
    test_scorce['Precision_scores'] = Metrics_list[3]
    test_scorce['Recall_scores'] = Metrics_list[4]
    test_scorce['F1_scores'] = Metrics_list[5]
    test_scorce['F2_scores'] = Metrics_list[6]
    metric_table_header = ["Metric_Name", "MEAN", "ET", "TC", "WT"]
    metric_table_left = ["Dice", "Jaccard", "Accuracy", "Precision", "Recall", "F1", "F2"]

    # 优化点：直接通过映射获取指标名称，避免重复字符串格式化

    
    metric_scores_mapping = {metric: test_scorce[f"{metric}_scores"] for metric in metric_table_left}
    metric_table = [[metric,
                    format_value(metric_scores_mapping[metric][0]),
                    format_value(metric_scores_mapping[metric][1]),
                    format_value(metric_scores_mapping[metric][2]),
                    format_value(metric_scores_mapping[metric][3])] for metric in metric_table_left]
    table_str = tabulate(metric_table, headers=metric_table_header, tablefmt='grid')
    metrics_info = table_str

    log_path = os.path.join(output_path, f"test_metrics.txt")
    custom_logger(metrics_info, log_path, log_time=True)
    print(metrics_info)

# def slide_window_pred(model, test_data, device, window_size, stride_size):
#     N, C, D, H, W = test_data.shape
#     model.eval()

#     with torch.no_grad():
#         with autocast(device_type='cuda'):
#             pred_mask = torch.zeros_like(test_data)
#             for d in range(0, D - window_size[0]+1, stride_size[0]): # D 维度
#                 for h in range(0, H - window_size[1]+1, stride_size[1]): # H 维度
#                     for w in range(0, W - window_size[2]+1, stride_size[2]): # W 维度
#                         patch = test_data[:, :, d:d+window_size[0], h:h+window_size[1], w:w+window_size[2]]
#                         patch = patch.to(device)

#                         pred = model(patch)

#                         # 将预测结果保存到原始图像的对应位置
#                         pred_mask[:, :, d:d+window_size[0], h:h+window_size[1], w:w+window_size[2]] = pred

#     return pred_mask

def slide_window_pred(model, test_data, device, window_size, stride_ratio=1):
# 改进自动计算stride的大小，并添加防止溢出的机制。
    N, C, D, H, W = test_data.shape
    model.eval()
    print(f"stride_ratio: {stride_ratio}")
    # 根据窗口大小和比例因子计算stride
    assert 0 < stride_ratio <= 1, "stride_ratio must be between 0 and 1"
    # stride_size = (int((D-window_size[0]) * stride_ratio), int((H - window_size[1]) * stride_ratio), int((W - window_size[2]) * stride_ratio))
    stride_size = (int(window_size[0] * stride_ratio), int(window_size[1] * stride_ratio), int(window_size[2] * stride_ratio))
    print(f"stride_size: {stride_size}")

    with torch.no_grad():
        with autocast(device_type='cuda'):  # 使用自动混合精度
            pred_mask = torch.zeros_like(test_data)
            # for d in range(0, D - window_size[0] +1, stride_size[0]):  # D 维度
            #     for h in range(0, H - window_size[1] +1, stride_size[1]):  # H 维度
            #         for w in range(0, W - window_size[2] +1, stride_size[2]):  # W 维度
            for d in range(0, D +1, stride_size[0]):  # D 维度
                for h in range(0, H +1, stride_size[1]):  # H 维度
                    for w in range(0, W +1, stride_size[2]):  # W 维度
                        if d + window_size[0] > D or h + window_size[1] > H or w + window_size[2] > W:
                            d = D - window_size[0]
                            h = H - window_size[1]
                            w = W - window_size[2]
                            end_d = D
                            end_h = H
                            end_w = W
                        else:
                            end_d = d + window_size[0]
                            end_h = h + window_size[1]
                            end_w = w + window_size[2]

                        # print(f"d:{d}, h:{h}, w:{w}")
                        # print(f"end_d:{end_d}, end_h:{end_h}, end_w:{end_w}")

                        patch = test_data[:, :, d:end_d, h:end_h, w:end_w]
                        patch = patch.to(device)

                        # print(f"patch.shape: {patch.shape}")
                        pred = model(patch)

                        # 将预测结果保存到原始图像的对应位置
                        pred_mask[:, :, d:end_d, h:end_h, w:end_w] = pred

    return pred_mask

def save_nii(pred_vimage, vimage, vmask, output_path, affine, i):

    # 降维，选出概率最大的类索引值
    test_output_argmax = torch.argmax(pred_vimage, dim=1).to(dtype=torch.int64) 
    num = 0

    # 获取测试集的输入图像和预测输出的图像数据
    save_input_t1 = vimage[num, 0, ...].permute(1, 2, 0).cpu().detach().numpy().astype(np.float32)
    save_input_t1ce = vimage[num, 1, ...].permute(1, 2, 0).cpu().detach().numpy().astype(np.float32)
    save_input_t2 = vimage[num, 2, ...].permute(1, 2, 0).cpu().detach().numpy().astype(np.float32)
    save_input_flair = vimage[num, 3, ...].permute(1, 2, 0).cpu().detach().numpy().astype(np.float32)
    save_input_mask = vmask[num, ...].permute(1, 2, 0).cpu().detach().numpy().astype(np.int8)
    save_pred = test_output_argmax[num, ...].permute(1, 2, 0).cpu().detach().numpy().astype(np.int8)

    # 将数据转成nib的对象
    nii_input_t1 = nib.nifti1.Nifti1Image(save_input_t1, affine=affine)
    nii_input_t1ce = nib.nifti1.Nifti1Image(save_input_t1ce, affine=affine)
    nii_input_t2 = nib.nifti1.Nifti1Image(save_input_t2, affine=affine)
    nii_input_flair = nib.nifti1.Nifti1Image(save_input_flair, affine=affine)
    nii_input_mask = nib.nifti1.Nifti1Image(save_input_mask, affine=affine)
    nii_pred = nib.nifti1.Nifti1Image(save_pred, affine=affine)

    output_path = os.path.join(output_path, f"P{i}")
    os.makedirs(output_path, exist_ok=True)

    # 保存nii文件
    nib.save(nii_input_t1, os.path.join(output_path, f'P{i}_test_input_t1.nii.gz'))
    nib.save(nii_input_t1ce, os.path.join(output_path, f'P{i}_test_input_t1ce.nii.gz'))
    nib.save(nii_input_t2, os.path.join(output_path, f'P{i}_test_input_t2.nii.gz'))
    nib.save(nii_input_flair, os.path.join(output_path, f'P{i}_test_input_flair.nii.gz'))
    nib.save(nii_input_mask, os.path.join(output_path, f'P{i}_test_input_mask.nii.gz'))
    nib.save(nii_pred, os.path.join(output_path, f'P{i}_test_pred.nii.gz'))

    print(f"P{i} pred save successfully! path on {output_path}")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_csv = args.test_csv
    bs, nw = 0, 0
    if args.data_scale == 'small':
        bs = 1
        nw = 4
    else:
        bs = 2
        nw = 8

    # 初始化模型
    if args.model == 'unet3d':
        model = UNET3D(4, 4, [32, 64, 128, 256])
    elif args.model == 'fm_unet3d_default':
        model = FM_UNET3D(in_channels=4, out_channels=4, ec_dilation_flags=[False, False, False, False], dc_dilation_flags=[False, False, False, False])
    elif args.model == 'fm_unet3d_res':
        model = FM_UNET3D(in_channels=4, out_channels=4, ec_dilation_flags=[False, False, False, False], dc_dilation_flags=[False, False, False, False], residuals_flag=True)
    elif args.model == 'fm_unet3d_dilation':
        model = FM_UNET3D(in_channels=4, out_channels=4, ec_dilation_flags=[True, False, False, False], dc_dilation_flags=[True, False, False, False])
    elif args.model == 'fm_unet3d_dilation_fusion':
        model = FM_UNET3D(in_channels=4, out_channels=4, fusion=args.fusion_flag, ec_dilation_flags=[True, False, False, False], dc_dilation_flags=[True, False, False, False])
    else:
        raise ValueError(f"Invalid model name: {args.model}")
    
    optimizer = AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.99), weight_decay=1e-5)
    scaler = GradScaler()
    Metricer = EvaluationMetrics()

    # 加载模型权重
    model, optimizer, scaler, start_epoch, best_val_loss = load_checkpoint(model, optimizer, scaler, args.ckpt_path)
    model.to(device)


    # 初始化数据集
    TransMethods_test   = data_transform(transform=Compose([RandomCrop3D(size=(154, 240, 240)),    # 随机裁剪
                                                        # tioRandonCrop3d(size=CropSize),
                                                        #  tioRandomFlip3d(),   
                                                         Normalize(mean=(0.114, 0.090, 0.170, 0.096), std=(0.199, 0.151, 0.282, 0.174)),   # 标准化
                                        ]))
    if args.data_scale == "full":
        test_dataset  = BraTS21_3d(test_csv, 
                                transform=TransMethods_test)
    elif args.data_scale == "small":
        test_dataset  = BraTS21_3d(test_csv,
                                transform=TransMethods_test,
                                local_train=True,
                                length=args.data_len)
    else:
        raise ValueError("data scale must be full or small")


    test_loader = DataLoader(dataset=test_dataset, 
                            batch_size=bs, 
                            num_workers=nw, 
                            shuffle=False)

    os.makedirs(args.outputs_root, exist_ok=True)

    affine = np.array([[ -1.,  -0.,  -0.,   0.],
                    [ -0.,  -1.,  -0., 239.],
                    [  0.,   0.,   1.,   0.],
                    [  0.,   0.,   0.,   1.]])
    # affine = -np.eye(4) #! 调整图像的方向


    dir_str = date_str + '_' +  os.path.splitext(os.path.basename(args.ckpt_path))[0]
    outputs_path = os.path.join(args.outputs_root, dir_str)

    inference(test_loader, model, Metricer, outputs_path, device, affine, save_flag=args.save_flag, stride_ratio=0.5)
    
    print("😃😃well done")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="inference args")

    parser.add_argument("--model", type=str,
                        default="f_cac_unet3d", 
                        help="model name")
    parser.add_argument("--data_scale", type=str, 
                        default="small", 
                        help="loading data scale")
    parser.add_argument("--data_len", type=int, 
                        default=4, 
                        help="train length")
    parser.add_argument("--test_csv", type=str, 
                        default="./brats21_local/test.csv", 
                        help="test csv file path")
    parser.add_argument("--ckpt_path", type=str, 
                        default=None, 
                        help='inference model path')
    parser.add_argument("--save_flag", type=bool, 
                        default=True, 
                        help="save flag")
    parser.add_argument("--outputs_root", type=str, 
                        default='./outputs', 
                        help="output path")
    args = parser.parse_args()

    main(args)