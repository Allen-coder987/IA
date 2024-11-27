import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNPP(nn.Module):
    def __init__(self):
        super(CNNPP, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, 6)  # 输出6个参数：gamma, wb_r, wb_g, wb_b, contrast, sharpen_strength
        )

    def forward(self, x):
        # 获取全局特征并通过CNN预测参数
        params = self.network(x)
        gamma = params[:, 0].clamp(0.8, 2.0)  # 控制伽马校正
        wb = params[:, 1:4].clamp(0.6, 1.5)  # 白平衡RGB值
        contrast = params[:, 4].clamp(0.8, 1.5)  # 对比度
        sharpen_strength = params[:, 5].clamp(0.0, 1.0)  # 锐化强度
        return gamma, wb, contrast, sharpen_strength
