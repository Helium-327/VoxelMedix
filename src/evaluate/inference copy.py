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

# 多进程
import multiprocessing
from multiprocessing import Pool, cpu_count

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

from datasets.BraTS21 import BraTS21_3D
from datasets.transforms import Compose, FrontGroundNormalize, RandomCrop3D, ToTensor
from loss_function import DiceLoss, CELoss
from metrics import *
from utils.logger_tools import custom_logger
from utils.ckpt_tools import load_checkpoint

from nnArchitecture.unet3d import UNet3D
from nnArchitecture.segFormer3d import SegFormer3D
from nnArchitecture.Mamba3d import Mamba3d
from nnArchitecture.MogaNet import MogaNet

# from utils.plot_tools.plot_results import NiiViewer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
model = UNet3D(in_channels=4, out_channels=4)
optimizer = AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.99), weight_decay=1e-5)
scaler = GradScaler()
Metricer = EvaluationMetrics()
Affine = np.array([[ -1.,  -0.,  -0.,   0.],
                [ -0.,  -1.,  -0., 239.],
                [  0.,   0.,   1.,   0.],
                [  0.,   0.,   0.,   1.]])
# affine = -np.eye(4) #! 调整图像的方向


test_datasets_paths = '/root/workspace/VoxelMedix/data/raw/brats21_original/test.csv'
test_trans = Compose([
    ToTensor(),
    RandomCrop3D(size=(155, 240, 240)),
    FrontGroundNormalize(),
    ])
test_dataset = BraTS21_3D(
    data_file=test_datasets_paths,
    transform=test_trans,
    local_train=True,
    length=10
    )

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=False   # 减少 worker 初始化时间
)

model_name = model.__class__.__name__
ckpt_path = '/root/workspace/VoxelMedix/results/2025-01-11/2025-01-11_22-04-46/checkpoints/best@e98_UNet3D__diceloss0.1403_dice0.8676_2025-01-11_22-04-46_13.pth'
output_path = os.path.join('/root/workspace/VoxelMedix/output', model_name)

def process_batch(data):
    try:
        # 处理数据
        pass
    except Exception as e:
        print(f"Error in subprocess: {e}")
    finally:
        torch.cuda.empty_cache()  # 释放 GPU 内存

def process_batch(vimage, vmask, model, device, window_size, stride_ratio):
    pred_vimage = slide_window_pred(model, vimage, device, window_size=window_size, stride_ratio=stride_ratio)
    return pred_vimage, vmask

def inference(
    test_loader=test_loader, 
    output_path=output_path, 
    model=model,
    optimizer=optimizer,
    scaler = scaler,
    metrics=Metricer,
    ckpt_path=ckpt_path,
    affine=Affine, 
    window_size=(128, 128, 128), 
    stride_ratio=0.5, 
    save_flag=True,
    device=DEVICE
    ):
    # 加载模型权重
    model, optimizer, scaler, start_epoch, best_val_loss = load_checkpoint(model, optimizer, scaler, ckpt_path)
    model.to(device)
    
    Metrics_list = np.zeros((7, 4))
    
    # 使用多进程池
    num_processes = cpu_count()  # 使用所有可用的 CPU 核心
    with Pool(processes=num_processes) as pool:
        results = []
        for data in tqdm(test_loader):
            results.append(pool.apply_async(process_batch, args=(data, model, device, window_size, stride_ratio)))
        
        pool.close()  # 关闭池，防止新任务提交
        pool.join()   # 等待所有子进程完成
        
        for i, result in enumerate(tqdm(results)):
            pred_vimage, vmask = result.get()
            
            # 保存预测结果nii文件
            os.makedirs(output_path, exist_ok=True)
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
    """在 3D 数据上执行滑动窗口预测。

    参数:
        model: 用于预测的模型。
        test_data: 输入数据张量。
        device: 运行模型的设备。
        window_size: 滑动窗口的大小。
        stride_ratio: 步幅与窗口大小的比例。

    返回:
        pred_mask: 预测的掩码。
    """
    N, C, D, H, W = test_data.shape
    model.eval()
    assert 0 < stride_ratio <= 1, "stride_ratio 必须在 0 和 1 之间"
    stride_size = (int(window_size[0] * stride_ratio),
                   int(window_size[1] * stride_ratio),
                   int(window_size[2] * stride_ratio))

    with torch.no_grad():
        with autocast(device_type='cuda'):
            pred_mask = torch.zeros_like(test_data, device=device)
            # count_mask = torch.zeros_like((N, model.out_channels, D, H, W), device=device)
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
                        # count_mask[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += 1

            # 避免除以零
            # count_mask[count_mask == 0] = 1
            # pred_mask /= count_mask
            pred_mask = pred_mask.cpu()

    return pred_mask

def save_nii(pred_vimage, vimage, vmask, output_path, affine, i):
    """将输入、掩码和预测输出保存为 NIfTI 文件。

    参数:
        pred_vimage: 预测的输出张量。
        vimage: 输入图像张量。
        vmask: 真实掩码张量。
        output_path: 保存文件的目录。
        affine: NIfTI 文件的仿射矩阵。
        i: 当前样本的索引。
    """
    test_output_argmax = torch.argmax(pred_vimage, dim=1).to(dtype=torch.int64)
    num = 0

    save_input_t1 = vimage[num, 0, ...].permute(1, 2, 0).cpu().detach().numpy().astype(np.float32)
    save_input_t1ce = vimage[num, 1, ...].permute(1, 2, 0).cpu().detach().numpy().astype(np.float32)
    save_input_t2 = vimage[num, 2, ...].permute(1, 2, 0).cpu().detach().numpy().astype(np.float32)
    save_input_flair = vimage[num, 3, ...].permute(1, 2, 0).cpu().detach().numpy().astype(np.float32)
    save_input_mask = vmask[num, ...].permute(1, 2, 0).cpu().detach().numpy().astype(np.int8)
    save_pred = test_output_argmax[num, ...].permute(1, 2, 0).cpu().detach().numpy().astype(np.int8)

    nii_input_t1 = nib.Nifti1Image(save_input_t1, affine=affine)
    nii_input_t1ce = nib.Nifti1Image(save_input_t1ce, affine=affine)
    nii_input_t2 = nib.Nifti1Image(save_input_t2, affine=affine)
    nii_input_flair = nib.Nifti1Image(save_input_flair, affine=affine)
    nii_input_mask = nib.Nifti1Image(save_input_mask, affine=affine)
    nii_pred = nib.Nifti1Image(save_pred, affine=affine)

    output_path = os.path.join(output_path, f"P{i}")
    os.makedirs(output_path, exist_ok=True)

    nib.save(nii_input_t1, os.path.join(output_path, f'P{i}_test_input_t1.nii.gz'))
    nib.save(nii_input_t1ce, os.path.join(output_path, f'P{i}_test_input_t1ce.nii.gz'))
    nib.save(nii_input_t2, os.path.join(output_path, f'P{i}_test_input_t2.nii.gz'))
    nib.save(nii_input_flair, os.path.join(output_path, f'P{i}_test_input_flair.nii.gz'))
    nib.save(nii_input_mask, os.path.join(output_path, f'P{i}_test_input_mask.nii.gz'))
    nib.save(nii_pred, os.path.join(output_path, f'P{i}_test_pred.nii.gz'))

    print(f"P{i} 预测结果保存成功！路径：{output_path}")


def main():
    inference(
    test_loader=test_loader, 
    output_path=output_path, 
    model=model,
    ckpt_path=ckpt_path,
    affine=Affine, 
    window_size=(128, 128, 128), 
    stride_ratio=0.5, 
    save_flag=True,
    device=DEVICE
    )
    
    print("😃😃well done")

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    

    # parser = argparse.ArgumentParser(description="inference args")

    # parser.add_argument("--model", type=str,
    #                     default="f_cac_unet3d", 
    #                     help="model name")
    # parser.add_argument("--data_scale", type=str, 
    #                     default="small", 
    #                     help="loading data scale")
    # parser.add_argument("--data_len", type=int, 
    #                     default=4, 
    #                     help="train length")
    # parser.add_argument("--test_csv", type=str, 
    #                     default="./brats21_local/test.csv", 
    #                     help="test csv file path")
    # parser.add_argument("--ckpt_path", type=str, 
    #                     default=None, 
    #                     help='inference model path')
    # parser.add_argument("--save_flag", type=bool, 
    #                     default=True, 
    #                     help="save flag")
    # parser.add_argument("--outputs_root", type=str, 
    #                     default='./outputs', 
    #                     help="output path")
    # args = parser.parse_args()

    main()
    