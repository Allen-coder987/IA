import torch
import torch.nn.functional as F
from modules.yolov5.utils.loss import ComputeLoss


def recovery_loss(enhanced, original):
    """计算恢复损失（MSE）"""
    mse_loss = F.mse_loss(enhanced, original)
    return mse_loss


def combined_loss(enhanced, original, model, targets, compute_loss):
    # 计算恢复损失（增强部分）
    rec_loss = recovery_loss(enhanced, original)

    # 前向传播
    pred = model(enhanced)  # 获取模型预测结果
    # 打印预测和标签的形状
    print("Prediction shape:", [p.shape for p in pred])
    print("Targets shape:", targets.shape)
    print("Targets:", targets)
    # 检查 targets 数据
    if (targets[:, 0] >= model.nc).any():
        raise ValueError(f"Class index out of range in targets: {targets[:, 0]}")
        # 如果目标框不多，扩展目标框以匹配多尺度的预测
    if targets.shape[1] == 1:  # 如果目标框数量为 1
        targets = targets.expand(len(pred), -1, -1)  # 扩展 targets 以匹配每个尺度
        # 检查 pred 和 targets 是否有形状匹配
    if targets.shape[0] != len(pred):
        raise ValueError(f"Shape mismatch between predictions and targets: {len(pred)} vs {targets.shape[0]}")

    # 使用 ComputeLoss 对象计算目标检测损失
    detection_loss, _ = compute_loss(pred, targets)

    # 返回联合损失
    total_loss = rec_loss + detection_loss
    return total_loss, rec_loss, detection_loss



# import torch
# import torch.nn.functional as F
# from torchmetrics.functional import structural_similarity_index_measure as ssim
# from modules.yolov5 import models
# from modules.yolov5.utils.loss import ComputeLoss
#
# def recovery_loss(enhanced, original):
#     """计算恢复损失（MSE + SSIM）"""
#     mse_loss = F.mse_loss(enhanced, original)
#     ssim_loss = 1 - ssim(enhanced, original)  # SSIM 越接近1越好
#     return mse_loss + ssim_loss
#
# # import torch
# # import torch.nn.functional as F
# # import pytorch_ssim  # 用于计算SSIM损失
# #
# # def recovery_loss(enhanced, original):
# #     """计算恢复损失（MSE + SSIM）"""
# #     mse_loss = F.mse_loss(enhanced, original)
# #     ssim_loss = 1 - pytorch_ssim.ssim(enhanced, original)  # SSIM 越接近1越好
# #     return mse_loss + ssim_loss
#
# def combined_loss(enhanced, original, model, targets, compute_loss):
#     # 计算恢复损失（图像增强部分）
#     rec_loss = recovery_loss(enhanced, original)
#
#     # 前向传播
#     pred = model(enhanced)  # 获取模型预测
#
#     # 使用已初始化的 ComputeLoss 对象计算目标检测损失
#     detection_loss, _ = compute_loss(pred, targets)
#
#     # 返回联合损失
#     total_loss = rec_loss + detection_loss
#     return total_loss, rec_loss, detection_loss

# def combined_loss(enhanced, original, model, targets):
#     """联合损失：增强损失 + YOLOv5目标检测损失"""
#     # 计算增强损失
#     rec_loss = recovery_loss(enhanced, original)
#     #通过模型进行前向传播
#     pred = model(enhanced) #获取模型的预测结果
#
#     # 计算YOLOv5检测损失
#     detection_loss = ComputeLoss(pred, targets)
#
#     # 总损失 = 增强损失 + 检测损失
#     total_loss = rec_loss + detection_loss
#     return total_loss, rec_loss, detection_loss
