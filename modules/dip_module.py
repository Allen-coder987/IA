import torch
import torch.nn as nn
import torch.nn.functional as F


class DIPModule(nn.Module):
    def __init__(self):
        super(DIPModule, self).__init__()

    def forward(self, x, gamma, wb, contrast, sharpen_strength):
        # Gamma校正
        x = x ** gamma.view(-1, 1, 1, 1)

        # 白平衡调整
        x = x * wb.view(-1, 3, 1, 1)

        # 对比度调整
        mean = torch.mean(x, dim=[2, 3], keepdim=True)
        x = torch.clamp((x - mean) * contrast.view(-1, 1, 1, 1) + mean, 0, 1)

        # 锐化处理
        sharpen_kernel = torch.tensor([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]],
                                      dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        # 应用卷积到每个通道
        x_sharpened = torch.cat([
            F.conv2d(x[:, i:i + 1], sharpen_kernel, padding=1) for i in range(x.shape[1])
        ], dim=1)

        # 混合原图与锐化图
        return torch.clamp(
            x_sharpened * sharpen_strength.view(-1, 1, 1, 1) + x * (1 - sharpen_strength.view(-1, 1, 1, 1)), 0, 1)



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class DIPModule(nn.Module):
#     def __init__(self):
#         super(DIPModule, self).__init__()
#
#     def forward(self, x, gamma, wb, contrast, sharpen_strength):
#         # Gamma校正
#         x = x ** gamma.view(-1, 1, 1, 1)  # 根据gamma调整图像亮度
#
#         # 白平衡调整
#         x = x * wb.view(-1, 3, 1, 1)  # 调整图像颜色通道
#
#         # 对比度调整
#         mean = torch.mean(x, dim=[2, 3], keepdim=True)  # 计算图像平均亮度
#         x = torch.clamp((x - mean) * contrast.view(-1, 1, 1, 1) + mean, 0, 1)  # 调整对比度
#
#         # 锐化处理
#         sharpen_kernel = torch.tensor([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]],
#                                       dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
#         x_sharpened = torch.cat([F.conv2d(x[:, i:i + 1], sharpen_kernel, padding=1) for i in range(3)], dim=1)
#         x = torch.clamp(x_sharpened * sharpen_strength.view(-1, 1, 1, 1) + x * (1 - sharpen_strength.view(-1, 1, 1, 1)), 0, 1)
#
#         return x
