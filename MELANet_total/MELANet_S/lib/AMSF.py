import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

class MonaOp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)

        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )

    def forward(self, x):
        identity = x
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)

        x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity

        identity = x

        x = self.projector(x)

        return identity + x

class AMSF(BaseModule):
    def __init__(self,
                 in_dim,
                 factor=4):
        super().__init__()

        # 改动1: 使用 Conv2d(kernel_size=1) 代替 Linear，直接处理 (B, C, H, W)
        self.project1 = nn.Conv2d(in_dim, 64, kernel_size=1)
        self.nonlinear = F.gelu
        self.project2 = nn.Conv2d(64, in_dim, kernel_size=1)

        self.dropout = nn.Dropout(p=0.1)

        self.adapter_conv = MonaOp(64)

        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))

    def forward(self, x):
        # x shape: (B, C, H, W)
        identity = x
        b, c, h, w = x.shape

        # 改动2: 适配 LayerNorm 和 参数广播
        # LayerNorm 通常作用于最后一维，所以需要 permute
        x_norm = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_norm = self.norm(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2) # (B, C, H, W)

   
        gamma = self.gamma.view(1, c, 1, 1)
        gammax = self.gammax.view(1, c, 1, 1)

        x = x_norm * gamma + x * gammax

    
        project1 = self.project1(x) # (B, 64, H, W)
        
    
        project1 = self.adapter_conv(project1) 

        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        
        project2 = self.project2(nonlinear) # (B, C, H, W)

        return identity + project2


if __name__ == '__main__':
    # 测试代码适配
    block = AMSF(in_dim=256).to('cuda')

    # 输入改为 (Batch, Channel, Height, Width)
    batch_size = 2
    h, w = 4, 4
    feature_dim = 256
    
    # 构造 B, C, H, W 格式的 tensor
    input_tensor = torch.rand(batch_size, feature_dim, h, w).to('cuda')

    # 不需要再传入 hw_shapes
    output = block(input_tensor)

    print("Input size (B, C, H, W):", input_tensor.size())
    print("Output size (B, C, H, W):", output.size())