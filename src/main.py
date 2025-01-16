# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/01/06 15:53:45
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 
*      VERSION: v1.0
=================================================
'''

import os
import json
from tracemalloc import start
import yaml

import torch
import torch.nn as nn
import argparse
from train import train
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, RMSprop, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.amp import GradScaler
from utils.logger_tools import *
from utils.shell_tools import *
from utils.tb_tools import *
from evaluate.metrics import EvaluationMetrics

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

from datasets.transforms import *
from datasets.BraTS21 import BraTS21_3D


from evaluate.lossFunc import *
from evaluate.metrics import *

# 环境设置
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
scaler = GradScaler()  # 混合精度训练
MetricsGo = EvaluationMetrics()  # 实例化评估指标类

def load_model(args):
    """加载模型"""
    if args.model_name == 'unet3d':
        model = UNet3D(in_channels=args.model_in_channel, out_channels=args.model_out_channel)
    elif args.model_name == 'soft_unet3d':
        model = soft_UNet3D(in_channels=args.model_in_channel, out_channels=args.model_out_channel)
    elif args.model_name == 'cad_unet3d':
        model = CAD_UNet3D(in_channels=args.model_in_channel, out_channels=args.model_out_channel)
    elif args.model_name == 'soft_cad_unet3d':
        model = soft_CAD_UNet3D(in_channels=args.model_in_channel, out_channels=args.model_out_channel)
    elif args.model_name == 'cadi_unet3d':
        model = CADI_UNet3D(in_channels=args.model_in_channel, out_channels=args.model_out_channel)
    elif args.model_name == 'soft_cadi_unet3d':
        model = soft_CADI_UNet3D(in_channels=args.model_in_channel, out_channels=args.model_out_channel)
    elif args.model_name == 'dw_unet3d':
        model = DW_UNet3D(in_channels=args.model_in_channel, out_channels=args.model_out_channel)
    elif args.model_name == 'soft_dw_unet3d':
        model = soft_DW_UNet3D(
            in_channels=args.model_in_channel, 
            out_channels=args.model_out_channel)
    elif args.model_name == 'segformer3d':
        model = SegFormer3D(
            in_channels=args.model_in_channel, 
            num_classes=args.model_out_channel
            )
    elif args.model_name == 'moga':
        model = MogaNet(
            in_channels=args.model_in_channel, 
            n_classes=args.model_out_channel
            )
    elif args.model_name == 'attention_unet':
        model = AttentionUnet(
            spatial_dims=3,
            in_channels=args.model_in_channel,
            out_channels=args.model_out_channel,
            channels=[32, 64, 128, 256, 320],
            strides=[2, 2, 2, 2],
        )
    elif args.model_name == 'mamba3d':#✅
        model = Mamba3d(
            in_channels=4, 
            n_classes=4
            )
    elif args.model_name == 'unetr': #✅
        model = UNETR(
            in_channels=args.model_in_channel,
            out_channels=args.model_out_channel,
            img_size=(128, 128, 128),
            feature_size=16,
            hidden_size=768,
            num_heads=12,
            spatial_dims=3,
            predict_mode=True  # 设置为预测模式
    )
    elif args.model_name == 'unetrpp':#✅
        model = UNETR_PP(
            in_channels=args.model_in_channel,
            out_channels=args.model_out_channel,  # 假设分割为2类
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
    elif args.model_name == 'uxnet': #✅
        model = UXNET(
            in_channels=args.model_in_channel, 
            out_channels=args.model_out_channel
            )
    else:
        raise ValueError(f"Incorrect input of parameter args.model_name:{args.model_name}, ")
    
    model = model.to(DEVICE)
    
    return model

def load_optimizer(args, model):
    """加载优化器"""
    if args.optimizer_type == 'adam':
        optimizer = Adam(model.parameters(), lr=float(args.optimizer_lr), betas=(0.9, 0.99), weight_decay=float(args.optimizer_wd))
    elif args.optimizer_type == 'sgd':
        optimizer = SGD(model.parameters(), lr=float(args.optimizer_lr), momentum=0.9, weight_decay=float(args.optimizer_wd))
    elif args.optimizer_type == 'rmsprop':
        optimizer = RMSprop(model.parameters(), lr=float(args.optimizer_lr), weight_decay=float(args.optimizer_wd))
    elif args.optimizer_type == 'adamw':
        optimizer = AdamW(model.parameters(), lr=float(args.optimizer_lr), betas=(0.9, 0.99), weight_decay=float(args.optimizer_wd))
    else:
        raise ValueError(f"Optimizer {args.optimizer_type} not supported")
    return optimizer

def load_scheduler(args, optimizer):
    """加载调度器"""
    if args.scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=float(args.factor), patience=int(args.patience), verbose=True)
    elif args.scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=int(args.scheduler_cosine_T_max), eta_min=float(args.scheduler_cosine_eta_min))
    elif args.scheduler_type == 'CosineWarmupRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=int(args.scheduler_cosine_T_0), T_mult=int(args.scheduler_cosine_T_mult), eta_min=float(args.scheduler_cosine_eta_min))
    else:
        raise ValueError(f"Scheduler {args.scheduler_type} not supported")
    return scheduler

def load_loss(args):
    """加载损失函数"""
    if args.loss_type == 'diceloss':
        loss_function = DiceLoss()
    elif args.loss_type == 'bceloss':
        loss_function = FocalLoss()
    elif args.loss_type == 'celoss':
        loss_function = CELoss()
    else:
        raise ValueError(f"Loss function {args.loss_type} not supported")
    return loss_function
    

def log_params(params, logs_path):
    """记录训练参数"""
    params_dict = {'Parameter': [str(p[0]) for p in list(params.items())],
                   'Value': [str(p[1]) for p in list(params.items())]}
    params_header = ["Parameter", "Value"]
    print(tabulate(params_dict, headers=params_header, tablefmt="grid"))
    custom_logger('='*40 + '\n' + "训练参数" +'\n' + '='*40 +'\n', logs_path, log_time=True)
    custom_logger(tabulate(params_dict, headers=params_header, tablefmt="grid"), logs_path)
    
def load_data(args):
    """加载数据集"""
    
    TransMethods_train = Compose([
        ToTensor(),
        RandomCrop3D(size=(128, 128, 128)),
        FrontGroundNormalize(),
        # tioRandomFlip3d(),
        # tioRandomElasticDeformation3d(),
        # tioZNormalization(),
        # tioRandomNoise3d(),
        # tioRandomGamma3d()
    ])

    TransMethods_val = Compose([
        ToTensor(),
        RandomCrop3D(size=(128, 128, 128)),
        FrontGroundNormalize(),
        # tioRandomFlip3d(),
        # tioRandomElasticDeformation3d(),
        # tioZNormalization(),
        # tioRandomNoise3d(),
        # tioRandomGamma3d()
    ])
    
    TransMethods_test = Compose([
        ToTensor(),
        RandomCrop3D(size=(155, 240, 240)),
        FrontGroundNormalize(),
        # tioRandomFlip3d(),
        # tioRandomElasticDeformation3d(),
        # tioZNormalization(),
        # tioRandomNoise3d(),
        # tioRandomGamma3d()
    ])

    train_dataset = BraTS21_3D(
        data_file=args.paths_train_csv,
        transform=TransMethods_train,
        local_train=args.training_local,
        length=args.training_train_length,
    )

    val_dataset = BraTS21_3D(
        data_file=args.paths_val_csv,
        transform=TransMethods_val,
        local_train=args.training_local,
        length=args.training_val_length,
    )

    test_dataset = BraTS21_3D(
        data_file=args.paths_test_csv,
        transform=TransMethods_test,
        local_train=args.training_local,
        length=args.training_test_length,
    )
    setattr(args, 'train_length', len(train_dataset))
    setattr(args, 'val_length', len(val_dataset))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.training_batch_size,
        shuffle=True,
        num_workers=args.training_num_workers,
        pin_memory=True,
        persistent_workers=True  # 减少 worker 初始化时间
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.training_batch_size,
        shuffle=False,
        num_workers=args.training_num_workers,
        pin_memory=True,
        persistent_workers=True  # 减少 worker 初始化时间
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.training_batch_size,
        shuffle=False,
        num_workers=args.training_num_workers,
        pin_memory=True,
        persistent_workers=True  # 减少 worker 初始化时间
    )
    
    print(f"已加载数据集, 训练集: {len(train_loader)}, 验证集: {len(val_loader)}")

    return train_loader, val_loader, test_loader

def main(args):
    start_epoch = 0
    best_val_loss = float('inf')
    resume_tb_path = None
    
    """------------------------------------- 模型实例化、初始化 --------------------------------------------"""
    # 加载模型
    model = load_model(args)
    # 加载优化器
    optimizer = load_optimizer(args, model)

    # 加载调度器
    scheduler = load_scheduler(args, optimizer)

    # 加载损失函数
    loss_function = load_loss(args)
    total_params = sum(p.numel() for p in model.parameters())
    total_params = f'{total_params/1024**2:.2f} M'
    print(f"Total number of parameters: {total_params}")
    setattr(args, 'total_parms', total_params)
    model_name = model.__class__.__name__
    # optimizer_name = optimizer.__class__.__name__
    # scheduler_name = scheduler.__class__.__name__
    # loss_name = loss_function.__class__.__name__
    
    """------------------------------------- 定义或获取路径 --------------------------------------------"""
    if args.paths_resume:
        resume_path = args.paths_resume
        print(f"Resuming training from {resume_path}")
        results_dir = os.path.join('/',*resume_path.split('/')[:-2])
        resume_tb_path = os.path.join(results_dir, 'tensorBoard')
        logs_dir = os.path.join(results_dir, 'logs')
        logs_file_name = [file for file in os.listdir(logs_dir) if file.endswith('.log')]
        logs_path = os.path.join(logs_dir, logs_file_name[0])
    else:
        os.makedirs(args.paths_results_root, exist_ok=True)
        results_dir = os.path.join(args.paths_results_root, ('_').join([model_name, f'{get_current_date()}_{get_current_time()}']))       # TODO: 改成网络对应的文件夹
        os.makedirs(results_dir, exist_ok=True)
        logs_dir = os.path.join(results_dir, 'logs')
        logs_path = os.path.join(logs_dir, f'{get_current_date()}.log')
        os.makedirs(logs_dir, exist_ok=True)

    """------------------------------------- 记录当前实验内容 --------------------------------------------"""
    exp_commit = args.commit if args.commit else input("请输入本次实验的更改内容: ")
    write_commit_file(os.path.join(results_dir, 'commits.md'), exp_commit)


    """------------------------------------- 断点续传 --------------------------------------------"""
    if args.paths_resume:
        print(f"Resuming training from checkpoint {args.paths_resume}")
        checkpoint = torch.load(args.paths_resume)
        best_val_loss = checkpoint['best_val_loss']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint {args.paths_resume}")
        print(f"Best val loss: {best_val_loss:.4f} ✈ epoch {start_epoch}")
        cutoff_tb_data(resume_tb_path, start_epoch)
        print(f"Refix resume tb data step {resume_tb_path} up to step {start_epoch}")

    # 加载数据集
    train_loader, val_loader, test_loader = load_data(args)


    # 记录训练配置
    log_params(vars(args), logs_path)

    """------------------------------------- 训练模型 --------------------------------------------"""
    train(model, 
          Metrics=MetricsGo, 
          train_loader=train_loader,
          val_loader=val_loader, 
          test_loader=test_loader,
          scaler=scaler, 
          optimizer=optimizer,
          scheduler=scheduler,
          loss_function=loss_function,
          num_epochs=args.training_epochs, 
          device=DEVICE, 
          results_dir=results_dir,
          logs_path=logs_path,
          output_path=args.paths_output,
          start_epoch=start_epoch,
          best_val_loss=best_val_loss,
          test_csv=args.paths_test_csv,
          tb=args.training_tb,
          interval=args.training_interval,
          save_max=args.training_save_max,
          early_stopping_patience=args.training_early_stop_patience,
          resume_tb_path=resume_tb_path)

def parse_args_from_yaml(yaml_file):
    """从 YAML 文件中解析配置参数"""
    assert os.path.exists(yaml_file), FileNotFoundError(f"Config file not found at {yaml_file}")
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def merge_args(local_args, global_args):
    """
    合并局部参数和全局参数，全局参数优先。
    """
    merged_args = local_args.copy()  # 复制局部参数
    for key, value in global_args.items():
        if value is not None:  # 只覆盖非空的全局参数
            if isinstance(value, dict) and key in merged_args:
                # 如果全局参数是字典且局部参数中也有该键，递归合并
                merged_args[key] = merge_args(merged_args[key], value)
            else:
                merged_args[key] = value
    return merged_args

def flatten_config(config, parent_key='', sep='_'):
    """
    将嵌套的配置字典展平为一层。
    """
    items = {}
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_config(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

if __name__ == '__main__':
    start_time = time.time()
    
    ## 定义全局参数
    parser = argparse.ArgumentParser(description='Train args')
    parser.add_argument('--config', type=str, 
                        default='/root/workspace/VoxelMedix/src/configs/2025_1_15/[GE]uxnet.yaml', 
                        help='Path to the configuration YAML file')
    parser.add_argument('--resume', type=str, 
                        default=False, 
                        help='Path to the checkpoint to resume training from')
    parser.add_argument('--resume_tb_path', type=str,
                        default=False, 
                        help='Path to the TensorBoard logs to resume from')
    parser.add_argument('--training_epochs', type=str,
                        default=100, help='training epoch')
    parser.add_argument('--training_train_length', type=str,
                        default=500, help='training epoch')
    parser.add_argument('--training_val_length', type=str,
                        default=100, help='training epoch')
    # parser.add_argument('--training_model', type=str,
    #                     default='unetr', help='training epoch')
    # 解析命令行参数
    global_args = vars(parser.parse_args())  

    # 2. 从 YAML 文件中加载局部参数
    local_args = parse_args_from_yaml(global_args['config'])

    # 3. 合并局部参数和全局参数，全局参数优先
    merged_args = merge_args(local_args, global_args)

    # 4. 展平配置字典（可选，如果需要扁平化参数）
    flattened_args = flatten_config(merged_args)

    end_time = time.time()
    
    print(f"加载配置文件耗时: {end_time - start_time:.2f} s")
    
    print(f"加载配置文件耗时: {end_time - start_time:.2f} s")
    main(args = argparse.Namespace(**flattened_args))