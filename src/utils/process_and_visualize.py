# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2025/01/14 22:11:40
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 可视化nii结果
*      VERSION: v1.0 
*      FEATURES: 可视化映射结果的单张slice, 可以调整slice_num 进行调整
=================================================
'''

import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def get_paths_dict(patient_dir: str) -> dict:
    """
    获取患者目录下的所有文件路径，并按模态分类。
    """
    paths_dict = {}
    filenames = os.listdir(patient_dir)
    for filename in filenames:
        if not filename.endswith('.nii.gz'):  # 确保文件是 NIfTI 格式
            continue
        modal = filename.split('.')[0].split('_')[-1]
        path = os.path.join(patient_dir, filename)
        paths_dict[modal] = path
    return paths_dict

def get_data_dict(paths_dict: dict) -> dict:
    """
    加载所有模态的数据。
    """
    data_dict = {}
    for modal, path in paths_dict.items():
        if not os.path.exists(path):  # 检查文件是否存在
            raise FileNotFoundError(f"File not found: {path}")
        data_ndarray = nib.load(path).get_fdata()
        data_dict[modal] = data_ndarray
    return data_dict

def get_slice_dict(data_dict: dict, slice_num: int) -> tuple:
    """
    提取指定切片的原始图像、真实掩膜和预测结果，并生成子区域的掩膜和预测结果。
    """
    sub_areas = ['ET', 'TC', 'WT']
    slices = {}
    subarea_slice_mask = {}
    subarea_slice_pred = {}

    # 提取原始图像、真实掩膜和预测结果
    slices['original'] = data_dict['t1'][:, :, slice_num]
    slices['mask'] = data_dict['mask'][:, :, slice_num]
    slices['pred'] = data_dict['pred'][:, :, slice_num]  # 修正：从 pred 中提取

    # 生成子区域的掩膜和预测结果
    for label, sub_area in enumerate(sub_areas):
        subarea_slice_mask[f'{sub_area} on Mask'] = np.where(slices['mask'] == label + 1, label + 1, 0)
        subarea_slice_pred[f'{sub_area} on Pred'] = np.where(slices['pred'] == label + 1, label + 1, 0)

    return slices, subarea_slice_mask, subarea_slice_pred

def _plot_slice(ax, original: np.ndarray, overlay: np.ndarray, title: str, cmap_overlay: str = 'cool', alpha: float = 0.5):
    """
    辅助函数：绘制原始图像和叠加的掩膜。
    """
    ax.imshow(original, cmap='bone')
    ax.imshow(np.ma.masked_where(overlay == 0, overlay), cmap=cmap_overlay, alpha=alpha)
    ax.set_title(title, fontsize=30)
    ax.axis('off')

def show_mapped_slice_mask_to_data(result_dir, slices: dict, addition: str):
    """
    显示真实掩膜和预测结果的对比图。
    """
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    
    # 绘制真实掩膜
    _plot_slice(axs[0], slices['original'], slices['mask'], 'Ground Truth')
    
    # 绘制预测结果
    _plot_slice(axs[1], slices['original'], slices['pred'], f'Prediction on {addition}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'{addition}_prediction.png'))
    plt.show()

def show_slice_dict(result_dir:str, single_slices: dict, mask_slice_dict: dict, cmap: str = 'bone', addition: str = 'UNet'):
    """
    显示多个子区域的分割结果。
    """
    fig, axs = plt.subplots(1, len(mask_slice_dict), figsize=(10 * len(mask_slice_dict), 10))
    
    for i, (title, mask_slice) in enumerate(mask_slice_dict.items()):
        _plot_slice(axs[i], single_slices['original'], mask_slice, f'{addition}: {title}', cmap_overlay='magma', alpha=1)
    
    plt.savefig(os.path.join(result_dir, f'{addition}_prediction.png'))
    plt.tight_layout()
    plt.show()

def process_and_visualize(patient_dir: str, slice_num: int, addition: str = 'UNet'):
    """
    串联所有函数，通过接受 NIfTI 文件路径直接输出结果。
    
    参数:
    patient_dir: 患者目录路径。
    slice_num: 要显示的切片编号。
    addition: 网络名称，用于标题显示。
    """
    # 1. 获取文件路径
    paths_dict = get_paths_dict(patient_dir)
    
    # 2. 加载数据
    data_dict = get_data_dict(paths_dict)
    
    # 3. 提取切片
    slices, subarea_slice_mask, subarea_slice_pred = get_slice_dict(data_dict, slice_num)
    
    # 4. 显示真实掩膜和预测结果的对比图
    show_mapped_slice_mask_to_data(patient_dir, slices, addition)
    
    # # 5. 显示多个子区域的分割结果
    # show_slice_dict(slices, subarea_slice_mask, addition=addition)
    # show_slice_dict(slices, subarea_slice_pred, addition=addition)


if __name__ == "__main__":
    # 示例调用
    patient_dir = '/root/workspace/VoxelMedix/output/UXNET/BraTS2021_00810'
    slice_num = 100  # 选择要显示的切片编号
    addition = 'UXNet'  # 网络名称

    process_and_visualize(patient_dir, slice_num, addition)