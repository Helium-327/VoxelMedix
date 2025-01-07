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
from nnArchitecture.unet3d import UNet3D, DW_UNet3D

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
    if args.model == 'unet3d':
        model = UNet3D(in_channels=args.in_channel, out_channels=args.out_channel)
    elif args.model == 'dw_unet3d':
        model = DW_UNet3D(in_channels=args.in_channel, out_channels=args.out_channel)
    model = model.to(DEVICE)
    
    return model

def load_optimizer(args, model):
    """加载优化器"""
    if args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=float(args.lr), betas=(0.9, 0.99), weight_decay=float(args.wd))
    elif args.optimizer == 'sgd':
        optimizer = SGD(model.parameters(), lr=float(args.lr), momentum=0.9, weight_decay=float(args.wd))
    elif args.optimizer == 'rmsprop':
        optimizer = RMSprop(model.parameters(), lr=float(args.lr), weight_decay=float(args.wd))
    elif args.optimizer == 'adamw':
        optimizer = AdamW(model.parameters(), lr=float(args.lr), betas=(0.9, 0.99), weight_decay=float(args.wd))
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")
    return optimizer

def load_scheduler(args, optimizer):
    """加载调度器"""
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=float(args.factor), patience=int(args.patience), verbose=True)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=int(args.cosine_T_max), eta_min=float(args.cosine_eta_min))
    elif args.scheduler == 'CosineWarmupRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=int(args.cosine_T_0), T_mult=int(args.cosine_T_mult), eta_min=float(args.cosine_eta_min))
    else:
        raise ValueError(f"Scheduler {args.scheduler} not supported")
    return scheduler

def load_loss(args):
    """加载损失函数"""
    if args.loss == 'diceloss':
        loss_function = DiceLoss()
    elif args.loss == 'bceloss':
        loss_function = FocalLoss()
    elif args.loss == 'celoss':
        loss_function = CELoss()
    else:
        raise ValueError(f"Loss function {args.loss} not supported")
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

    train_dataset = BraTS21_3D(
        data_file=args.train_csv,
        transform=TransMethods_train,
        local_train=args.local_train,
        length=args.train_length,
    )

    val_dataset = BraTS21_3D(
        data_file=args.val_csv,
        transform=TransMethods_val,
        local_train=args.local_train,
        length=args.val_length,
    )

    setattr(args, 'train_length', len(train_dataset))
    setattr(args, 'val_length', len(val_dataset))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True  # 减少 worker 初始化时间
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True  # 减少 worker 初始化时间
    )
    
    print(f"已加载数据集, 训练集: {len(train_loader)}, 验证集: {len(val_loader)}")

    return train_loader, val_loader

def main(args):
    start_epoch = 0
    best_val_loss = float('inf')
    resume_tb_path = None

    """------------------------------------- 定义或获取路径 --------------------------------------------"""
    if args.resume:
        resume_path = args.resume
        print(f"Resuming training from {resume_path}")
        results_dir = os.path.join(*resume_path.split('/')[:-2])
        resume_tb_path = os.path.join(results_dir, 'tensorBoard')
        logs_dir = os.path.join(results_dir, 'logs')
        logs_file_name = [file for file in os.listdir(logs_dir) if file.endswith('.log')]
        logs_path = os.path.join(logs_dir, logs_file_name[0])
    else:
        os.makedirs(args.results_root, exist_ok=True)
        results_dir = os.path.join(args.results_root, get_current_date())
        results_dir = create_folder(results_dir)
        logs_dir = os.path.join(results_dir, 'logs')
        logs_path = os.path.join(logs_dir, f'{get_current_date()}.log')
        os.makedirs(logs_dir, exist_ok=True)

    """------------------------------------- 记录当前实验内容 --------------------------------------------"""
    exp_commit = args.commit if args.commit else input("请输入本次实验的更改内容: ")
    write_commit_file(os.path.join(results_dir, 'commits.md'), exp_commit)

    """------------------------------------- 模型实例化、初始化 --------------------------------------------"""
    model = load_model(args)
    total_params = sum(p.numel() for p in model.parameters())
    total_params = f'{total_params/1024**2:.2f} M'
    print(f"Total number of parameters: {total_params}")
    setattr(args, 'total_parms', total_params)

    """------------------------------------- 断点续传 --------------------------------------------"""
    if args.resume:
        print(f"Resuming training from checkpoint {args.resume}")
        checkpoint = torch.load(args.resume)
        best_val_loss = checkpoint['best_val_loss']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint {args.resume}")
        print(f"Best val loss: {best_val_loss:.4f} ✈ epoch {start_epoch}")
        cutoff_tb_data(resume_tb_path, start_epoch)
        print(f"Refix resume tb data step {resume_tb_path} up to step {start_epoch}")

    # 加载数据集
    train_loader, val_loader = load_data(args)

    # 加载优化器
    optimizer = load_optimizer(args, model)

    # 加载调度器
    scheduler = load_scheduler(args, optimizer)

    # 加载损失函数
    loss_function = load_loss(args)

    # 记录训练配置
    log_params(vars(args), logs_path)

    """------------------------------------- 训练模型 --------------------------------------------"""
    train(model, 
          Metrics=MetricsGo, 
          train_loader=train_loader,
          val_loader=val_loader, 
          scaler=scaler, 
          optimizer=optimizer,
          scheduler=scheduler,
          loss_function=loss_function,
          num_epochs=args.epochs, 
          device=DEVICE, 
          results_dir=results_dir,
          logs_path=logs_path,
          start_epoch=start_epoch,
          best_val_loss=best_val_loss,
          tb=args.tb,
          interval=args.interval,
          save_max=args.save_max,
          early_stopping_patience=args.early_stop_patience,
          resume_tb_path=resume_tb_path)

def parse_args_from_yaml(yaml_file):
    """从 YAML 文件中解析配置参数"""
    assert os.path.exists(yaml_file), FileNotFoundError(f"Config file not found at {yaml_file}")
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Train args')
    parser.add_argument('--config', type=str, 
                        default='/root/workspace/VoxelMedix/src/configs/dw_unet3d.yaml', 
                        help='Path to the configuration YAML file')
    parser.add_argument('--resume', type=str, 
                        default=None, 
                        help='Path to the checkpoint to resume training from')
    parser.add_argument('--resume_tb_path', type=str,
                        default=None, 
                        help='Path to the TensorBoard logs to resume from')
    
    args = parser.parse_args()
    args = argparse.Namespace(**parse_args_from_yaml(args.config))
    end_time = time.time()
    
    print(f"加载配置文件耗时: {end_time - start_time:.2f} s")
    main(args=args)