# -*- coding: UTF-8 -*-
'''

Describle:         模型评估：在训练完成之后使用测试集对模型性能进行评估，并保存评估结果

Created on         2024/08/18 14:07:26
Author:            @ Mr_Robot
Current State:     #TODO:
'''
from genericpath import exists
import sys
sys.path.append('/root/workspace/VoxelMedix/src')

# # 多进程
# import multiprocessing
# from multiprocessing import Pool, cpu_count
import pandas as pd
import os
import torch
import shutil
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

from datasets.BraTS21 import BraTS21_3D
from datasets.transforms import Compose, FrontGroundNormalize, RandomCrop3D, ToTensor
from loss_function import DiceLoss, CELoss
from evaluate.metrics import *
from utils.logger_tools import custom_logger, get_current_date, get_current_time
from utils.ckpt_tools import load_checkpoint

from nnArchitecture.unet3d import *
from nnArchitecture.uxnet import UXNET
from nnArchitecture.segFormer3d import SegFormer3D
from nnArchitecture.MogaNet import MogaNet
from nnArchitecture.AtentionUNet import AttentionUnet
from nnArchitecture.Mamba3d import Mamba3d
from nnArchitecture.unetr import UNETR
from nnArchitecture.unetrpp import UNETR_PP
from nnArchitecture.SwinUNETRv2 import SwinUNETR
# from nnArchitecture.dw_unet3d import  DW_UNet3D

# from utils.plot_tools.plot_results import NiiViewer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def load_model(model_name, in_channels=4, out_channels=4):
    """加载模型"""
    if model_name == 'unet3d':
        model = UNet3D(in_channels=in_channels, out_channels=out_channels)
    elif model_name == 'soft_unet3d':
        model = soft_UNet3D(in_channels=in_channels, out_channels=out_channels)
    elif model_name == 'cad_unet3d':
        model = CAD_UNet3D(in_channels=in_channels, out_channels=out_channels)
    elif model_name == 'soft_cad_unet3d':
        model = soft_CAD_UNet3D(in_channels=in_channels, out_channels=out_channels)
    elif model_name == 'cadi_unet3d':
        model = CADI_UNet3D(in_channels=in_channels, out_channels=out_channels)
    elif model_name == 'soft_cadi_unet3d':
        model = soft_CADI_UNet3D(in_channels=in_channels, out_channels=out_channels)
    elif model_name == 'dw_unet3d':
        model = DW_UNet3D(in_channels=in_channels, out_channels=out_channels)
    elif model_name == 'soft_dw_unet3d':
        model = soft_DW_UNet3D(in_channels=in_channels, out_channels=out_channels)
    elif model_name == 'segformer3d':
        model = SegFormer3D(in_channels=in_channels, num_classes=out_channels)
    elif model_name == 'moga':
        model = MogaNet(in_channels=in_channels, n_classes=out_channels)
    elif model_name == 'attention_unet':
        model = AttentionUnet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=[32, 64, 128, 256, 320],
            strides=[2, 2, 2, 2],
        )
    elif model_name == 'mamba3d':
        model = Mamba3d(in_channels=in_channels, n_classes=out_channels)
    elif model_name == 'unetr':
        model = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=(128, 128, 128),
            feature_size=16,
            hidden_size=768,
            num_heads=12,
            spatial_dims=3,
            predict_mode=True  # 设置为预测模式
        )
    elif model_name == 'unetrpp':
        model = UNETR_PP(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=16,
            hidden_size=256,
            num_heads=8,
            pos_embed="perceptron",
            norm_name="instance",
            dropout_rate=0.1,
            depths=[3, 3, 3, 3],
            dims=[32, 64, 128, 256],
            conv_op=nn.Conv3d,
            do_ds=False,
        )
    elif model_name == 'uxnet':
        model = UXNET(in_channels=in_channels, out_channels=out_channels)
    else:
        raise ValueError(f"Incorrect input of parameter model_name:{model_name}")
    
    model = model.to(DEVICE)
    return model



def load_data(test_csv, local_train=True, test_length=10, batch_size=1, num_workers=4):
    """加载数据集"""
    TransMethods_test = Compose([
        ToTensor(),
        RandomCrop3D(size=(155, 240, 240)),
        FrontGroundNormalize(),
    ])

    test_dataset = BraTS21_3D(
        data_file=test_csv,
        transform=TransMethods_test,
        local_train=local_train,
        length=test_length,
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True  # 减少 worker 初始化时间
    )
    
    print(f"已加载测试数据: {len(test_loader)}")
    return test_loader

def inference(test_df, test_loader, output_path, model, Metricer, scaler, optimizer, ckpt_path, affine=None, window_size=(128, 128, 128), stride_ratio=0.5, save_flag=True, device=DEVICE):
    if affine is None:
        affine = np.array([[ -1.,  -0.,  -0.,   0.],
                           [ -0.,  -1.,  -0., 239.],
                           [  0.,   0.,   1.,   0.],
                           [  0.,   0.,   0.,   1.]])
    
    # 加载模型权重
    model, optimizer, scaler, _, _ = load_checkpoint(model, optimizer, scaler, ckpt_path)
    model.to(device)
    
    Metrics_list = np.zeros((7, 4))
        
    for i, data in enumerate(tqdm(test_loader)):
        vimage, vmask = data[0], data[1]
        pred_vimage = slide_window_pred(model, vimage, device, window_size, stride_ratio=1)
        
        # 获取病例号
        case_id = test_df.iloc[i]['patient_idx']
        
        # 保存预测结果nii文件
        os.makedirs(output_path, exist_ok=True)
        if save_flag:
            save_nii(test_df, pred_vimage, vmask, output_path, affine, case_id)
        
        # 评估指标
        metrics = Metricer.update(pred_vimage, vmask)
        Metrics_list += metrics

    Metrics_list /= len(test_loader)

    test_scorce = {}
    test_scorce['Dice_scores'] = Metrics_list[0] 
    test_scorce['Jaccard_scores'] = Metrics_list[1]
    test_scorce['Accuracy_scores'] = Metrics_list[2]
    test_scorce['Precision_scores'] = Metrics_list[3]
    test_scorce['Recall_scores'] = Metrics_list[4]
    test_scorce['F1_scores'] = Metrics_list[5]
    test_scorce['F2_scores'] = Metrics_list[6]
    
    metric_table_header = ["Metric_Name", "MEAN", "ET", "TC", "WT"]
    metric_table_left = ["Dice", "Jaccard", "Accuracy", "Precision", "Recall", "F1", "F2"]

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

def slide_window_pred(model, test_data, device, window_size, stride_ratio=1):
    """在 3D 数据上执行滑动窗口预测。"""
    N, C, D, H, W = test_data.shape
    model.eval()
    assert 0 < stride_ratio <= 1, "stride_ratio 必须在 0 和 1 之间"
    stride_size = (int(window_size[0] * stride_ratio),
                   int(window_size[1] * stride_ratio),
                   int(window_size[2] * stride_ratio))

    with torch.no_grad():
        with autocast(device_type='cuda'):
            pred_mask = torch.zeros_like(test_data, device=device)
            for d in range(0, D, stride_size[0]):
                for h in range(0, H, stride_size[1]):
                    for w in range(0, W, stride_size[2]):
                        d_start = min(d, D - window_size[0])
                        h_start = min(h, H - window_size[1])
                        w_start = min(w, W - window_size[2])
                        d_end = d_start + window_size[0]
                        h_end = h_start + window_size[1]
                        w_end = w_start + window_size[2]

                        patch = test_data[:, :, d_start:d_end, h_start:h_end, w_start:w_end].to(device)
                        pred = model(patch)
                        if isinstance(pred, list):
                            pred = pred[-1]  # 使用最后一个预测结果
                        pred = pred.float()  # 确保预测结果是浮点型

                        pred_mask[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += pred

            pred_mask = pred_mask.cpu()

    return pred_mask

def save_nii(test_df, pred_vimage, vmask, output_path, affine, case_id):
    """将输入、掩码和预测输出保存为 NIfTI 文件。"""
    test_output_argmax = torch.argmax(pred_vimage, dim=1).to(dtype=torch.int64)
    num = 0

    save_input_mask = vmask[num, ...].permute(1, 2, 0).cpu().detach().numpy().astype(np.int8)
    save_pred = test_output_argmax[num, ...].permute(1, 2, 0).cpu().detach().numpy().astype(np.int8)

    nii_input_mask = nib.Nifti1Image(save_input_mask, affine=affine)
    nii_pred = nib.Nifti1Image(save_pred, affine=affine)

    output_path = os.path.join(output_path, f"{case_id}")
    os.makedirs(output_path, exist_ok=True)

    nib.save(nii_input_mask, os.path.join(output_path, f'{case_id}_test_input_mask.nii.gz'))
    nib.save(nii_pred, os.path.join(output_path, f'{case_id}_test_pred.nii.gz'))
    
    case_data_path = test_df.loc[test_df['patient_idx'] == case_id, 'patient_dir'].values[0]
    for file_name in os.listdir(case_data_path):
        src_file = os.path.join(case_data_path, file_name)
        dst_file = os.path.join(output_path, file_name)
        shutil.copy(src_file, dst_file)
        
    print(f"{case_id} 预测结果保存成功！路径：{output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with UXNET model.")
    parser.add_argument('--model_name', type=str, default='mamba3d', help='Model name (default: uxnet)')
    parser.add_argument('--test_csv', type=str, default='/root/workspace/VoxelMedix/data/raw/brats21_original/test.csv', help='Path to the test dataset CSV file (default: /root/workspace/VoxelMedix/data/raw/brats21_original/test.csv)')
    parser.add_argument('--ckpt_path', type=str, default='/root/workspace/VoxelMedix/results/Mamba3d_DiceLoss_AdamW_CosineAnnealingWarmRestarts/2025-01-15_03-01-11/checkpoints/best@e12_Mamba3d__diceloss0.2657_dice0.7270_2025-01-15_03-01-11_5.pth', help='Path to the model checkpoint file (default: /root/workspace/VoxelMedix/results/UXNET_DiceLoss_AdamW_CosineAnnealingWarmRestarts/2025-01-14_22-12-52/checkpoints/UXNET_final_model.pth)')
    parser.add_argument('--output_path', type=str, default='/root/workspace/VoxelMedix/output/Mamba3d', help='Output directory to save results (default: /root/workspace/VoxelMedix/output/UXNET)')
    return parser.parse_args()

def main():
    args = parse_args()
    model = load_model(args.model_name)
    test_loader = load_data(args.test_csv)
    test_df = pd.read_csv(args.test_csv)
    Metricer = EvaluationMetrics()
    scaler = GradScaler()
    optimizer = AdamW(model.parameters(), lr=0.0002, betas=(0.9, 0.99), weight_decay=0.00001)
    inference(test_df, test_loader, args.output_path, model, Metricer, scaler, optimizer, args.ckpt_path)
    print("😃😃 Well done!")

if __name__ == '__main__':
    main()