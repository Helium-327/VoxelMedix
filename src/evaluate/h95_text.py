import torch
import torch.nn.functional as F

# def extract_boundary(mask):
#     """
#     提取三维掩码的边界点集。
    
#     参数:
#     - mask: 形状为 (D, H, W) 的掩码张量。
    
#     返回:
#     - boundary_points: 边界点集的坐标，形状为 (N, 3)。
#     """
#     # 定义一个 3D 卷积核，用于检测边界
#     kernel = torch.ones(1, 1, 3, 3, 3).to(mask.device)  # 3x3x3 的卷积核
    
#     # 对掩码进行卷积
#     mask = mask.unsqueeze(0).unsqueeze(0).float()  # 形状: (1, 1, D, H, W)
#     conv_result = F.conv3d(mask, kernel, padding=1)  # 形状: (1, 1, D, H, W)
    
#     # 边界点是那些卷积结果不等于 27 * mask 的点
#     boundary = (conv_result != 27 * mask).squeeze()  # 形状: (D, H, W)
    
#     # 获取边界点的坐标
#     boundary_points = torch.nonzero(boundary).float()  # 形状: (N, 3)
    
#     return boundary_points

# def hausdorff_distance(pred_boundary, true_boundary, batch_size=1000):
#     """
#     计算两组边界点集之间的豪斯多夫距离。
    
#     参数:
#     - pred_boundary: 预测的边界点集，形状为 (N1, 3)。
#     - true_boundary: 真实的边界点集，形状为 (N2, 3)。
#     - batch_size: 每批的点集大小。
    
#     返回:
#     - hausdorff_dist: 豪斯多夫距离。
#     """
#     max_min_pred_to_true = 0
#     max_min_true_to_pred = 0
    
#     # 分批处理 pred_points
#     for i in range(0, pred_boundary.shape[0], batch_size):
#         pred_batch = pred_boundary[i:i + batch_size]  # 形状: (batch_size, 3)
#         distances = torch.cdist(pred_batch, true_boundary)  # 形状: (batch_size, N2)
#         min_distances_pred_to_true = torch.min(distances, dim=1).values  # 形状: (batch_size,)
#         max_min_pred_to_true = max(max_min_pred_to_true, torch.max(min_distances_pred_to_true).item())
    
#     # 分批处理 true_points
#     for j in range(0, true_boundary.shape[0], batch_size):
#         true_batch = true_boundary[j:j + batch_size]  # 形状: (batch_size, 3)
#         distances = torch.cdist(true_batch, pred_boundary)  # 形状: (batch_size, N1)
#         min_distances_true_to_pred = torch.min(distances, dim=1).values  # 形状: (batch_size,)
#         max_min_true_to_pred = max(max_min_true_to_pred, torch.max(min_distances_true_to_pred).item())
    
#     # 豪斯多夫距离是两者的最大值
#     hausdorff_dist = max(max_min_pred_to_true, max_min_true_to_pred)
    
#     return hausdorff_dist

import torch

def hausdorff_distance_gpu(seg1, seg2):
    border1 = torch.nonzero(torch.tensor(seg1)).float().cuda()
    border2 = torch.nonzero(torch.tensor(seg2)).float().cuda()
    
    # 计算距离矩阵
    distances = torch.cdist(border1, border2)
    
    hd1 = torch.max(torch.min(distances, dim=1).values)
    hd2 = torch.max(torch.min(distances, dim=0).values)
    
    return max(hd1.item(), hd2.item())


if __name__ == '__main__':
    # 示例掩码
    pred_mask = torch.rand((128, 128, 128), device='cuda')  # 形状: [128, 128, 128]
    true_mask = torch.randint(0, 2, (128, 128, 128), device='cuda')  # 形状: [128, 128, 128]

    # # 提取边界点集
    # pred_boundary = extract_boundary(pred_mask)  # 形状: (N1, 3)
    # true_boundary = extract_boundary(true_mask)  # 形状: (N2, 3)

    # # 计算豪斯多夫距离
    # hausdorff_dist = hausdorff_distance(pred_boundary, true_boundary)

    # print(hausdorff_dist)
    hd = hausdorff_distance_gpu(pred_mask, true_mask)
    print(hd)