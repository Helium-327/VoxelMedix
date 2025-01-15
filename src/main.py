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
    if args.model == 'unet3d':
        model = UNet3D(in_channels=args.in_channel, out_channels=args.out_channel)
    elif args.model == 'soft_unet3d':
        model = soft_UNet3D(in_channels=args.in_channel, out_channels=args.out_channel)
    elif args.model == 'cad_unet3d':
        model = CAD_UNet3D(in_channels=args.in_channel, out_channels=args.out_channel)
    elif args.model == 'soft_cad_unet3d':
        model = soft_CAD_UNet3D(in_channels=args.in_channel, out_channels=args.out_channel)
    elif args.model == 'cadi_unet3d':
        model = CADI_UNet3D(in_channels=args.in_channel, out_channels=args.out_channel)
    elif args.model == 'soft_cadi_unet3d':
        model = soft_CADI_UNet3D(in_channels=args.in_channel, out_channels=args.out_channel)
    elif args.model == 'dw_unet3d':
        model = DW_UNet3D(in_channels=args.in_channel, out_channels=args.out_channel)
    elif args.model == 'soft_dw__unet3d':
        model = soft_DW_UNet3D(
            in_channels=args.in_channel, 
            out_channels=args.out_channel)
    elif args.model == 'segformer3d':
        model = SegFormer3D(
            in_channels=args.in_channel, 
            num_classes=args.out_channel
            )
    elif args.model == 'moga':
        model = MogaNet(
            in_channels=args.in_channel, 
            n_classes=args.out_channel
            )
    elif args.model == 'attention_unet':
        model = AttentionUnet(
            spatial_dims=3,
            in_channels=args.in_channel,
            out_channels=args.out_channel,
            channels=[32, 64, 128, 256, 320],
            strides=[2, 2, 2, 2],
        )
    elif args.model == 'mamba3d':#✅
        model = Mamba3d(
            in_channels=4, 
            n_classes=4
            )
    elif args.model == 'unetr': #✅
        model = UNETR(
            in_channels=args.in_channel,
            out_channels=args.out_channel,
            img_size=(128, 128, 128),
            feature_size=16,
            hidden_size=768,
            num_heads=12,
            spatial_dims=3,
            predict_mode=True  # 设置为预测模式
    )
    elif args.model == 'unetrpp':#✅
        model = UNETR_PP(
            in_channels=args.in_channel,
            out_channels=args.out_channel,  # 假设分割为2类
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
    elif args.model == 'uxnet': #✅
        model = UXNET(
            in_channels=args.in_channel, 
            out_channels=args.out_channel
            )
    else:
        raise ValueError(f"Incorrect input of parameter args.model:{args.model}, ")
    
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

    test_dataset = BraTS21_3D(
        data_file=args.test_csv,
        transform=TransMethods_test,
        local_train=args.local_train,
        length=args.test_length,
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
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
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
    optimizer_name = optimizer.__class__.__name__
    scheduler_name = scheduler.__class__.__name__
    loss_name = loss_function.__class__.__name__
    
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
        results_dir = os.path.join(args.results_root, ('_').join([model_name, loss_name, optimizer_name, scheduler_name]))       # TODO: 改成网络对应的文件夹
        results_dir = create_folder(results_dir)
        logs_dir = os.path.join(results_dir, 'logs')
        logs_path = os.path.join(logs_dir, f'{get_current_date()}.log')
        os.makedirs(logs_dir, exist_ok=True)

    """------------------------------------- 记录当前实验内容 --------------------------------------------"""
    exp_commit = args.commit if args.commit else input("请输入本次实验的更改内容: ")
    write_commit_file(os.path.join(results_dir, 'commits.md'), exp_commit)


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
          num_epochs=args.epochs, 
          device=DEVICE, 
          results_dir=results_dir,
          logs_path=logs_path,
          output_path=args.output_path,
          start_epoch=start_epoch,
          best_val_loss=best_val_loss,
          test_csv=args.test_csv,
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
                        default='/root/workspace/VoxelMedix/src/configs/debug.yaml', 
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