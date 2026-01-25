# import torch
# import torch.nn as nn
# from einops import rearrange
#
#
# class RFA(nn.Module):
#     def __init__(self, dim, num_heads=4, bias=False):
#         super(RFA, self).__init__()
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
#
#         self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
#         self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
#         self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#         qkv = self.qkv_dwconv(self.qkv(x))
#         q, k, v = qkv.chunk(3, dim=1)
#
#         q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#
#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)
#
#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#         attn = attn.softmax(dim=-1)
#
#         out = (attn @ v)
#         out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
#         out = self.project_out(out)
#         return out + x
#
#
#
#
#
# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         return x * self.fc(y).view(b, c, 1, 1).expand_as(x)
#
#
# class SFF(nn.Module):
#     def __init__(self, in_channels_x, in_channels_y):
#         super(SFF, self).__init__()
#         self.channel_adjust_y = nn.Sequential(
#             nn.Conv2d(in_channels_y, in_channels_x, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(in_channels_x),
#             nn.ReLU(inplace=True)
#         )
#         self.channel_adjust_x = nn.Sequential(
#             nn.Conv2d(in_channels_x, in_channels_x, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(in_channels_x),
#             nn.ReLU(inplace=True)
#         )
#         self.se_x = SELayer(in_channels_x)
#         self.se_y = SELayer(in_channels_y)
#         self.rf = RFA(in_channels_x)
#
#
#     def forward(self, x, y):
#
#         x_se = self.se_x(x)
#         x_se = self.channel_adjust_x(x_se)
#         y_se = self.se_y(y)
#         y_se = self.channel_adjust_y(y_se)
#         z_0 = x_se + y_se
#         z_1 = self.rf(z_0)
#
#         return z_1 + x
#
#
# if __name__ == '__main__':
#     # 测试：x和y尺寸相同（128x128），通道数不同（16和32）
#     x = torch.randn(4, 16, 128, 128)  # 输入x：(4, 16, 128, 128)
#     y = torch.randn(4, 32, 128, 128)  # 输入y：(4, 32, 128, 128)
#
#     model = SFF(in_channels_x=16, in_channels_y=32)
#     out = model(x, y)
#
#     print(f"输入x形状: {x.shape}")  # torch.Size([4, 16, 128, 128])
#     print(f"输入y形状: {y.shape}")  # torch.Size([4, 32, 128, 128])
#     print(f"输出形状: {out.shape}")  # torch.Size([4, 16, 128, 128])（与ximport torch
import torch.nn as nn
from einops import rearrange

class RFA(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(RFA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out + x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(channel // reduction, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channel // reduction, 1), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        return x * self.fc(y).view(b, c, 1, 1).expand_as(x)


class SFF(nn.Module):
    """
    Enhanced SFF:
      - channel_adjust_x/y: 1x1 conv + BN + ReLU (keep lightweight)
      - SE on both branches (as before)
      - learnable alpha to weight y contribution: z0 = x_se + alpha * y_se
      - RFA for spatial modeling
      - fusion_conv (3x3) on concat([x, z1]) to produce final feature of same channels as x
      - residual connection: out = fusion_out + x
    """
    def __init__(self, in_channels_x, in_channels_y, dropout=0.0):
        super(SFF, self).__init__()

        # channel adjust (keep 1x1 convs as channel projectors)
        self.channel_adjust_y = nn.Sequential(
            nn.Conv2d(in_channels_y, in_channels_x, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels_x),
            nn.ReLU(inplace=True)
        )
        self.channel_adjust_x = nn.Sequential(
            nn.Conv2d(in_channels_x, in_channels_x, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels_x),
            nn.ReLU(inplace=True)
        )

        # SE blocks
        self.se_x = SELayer(in_channels_x)
        self.se_y = SELayer(in_channels_y)

        # RFA for spatial modelling (keeps channel = in_channels_x)
        self.rf = RFA(in_channels_x)

        # learnable weight for y contribution (scalar)
        self.alpha = nn.Parameter(torch.tensor(1.0))  # initialize to 1.0 (equal contribution initially)

        # fusion conv: concat(x, z1) -> reduce back to in_channels_x
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels_x * 2, in_channels_x, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels_x),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        )

    def forward(self, x, y):
        """
        x: tensor with shape (B, in_channels_x, H, W)
        y: tensor with shape (B, in_channels_y, H, W)
        returns: tensor with shape (B, in_channels_x, H, W)
        """
        # SE attention
        x_se = self.se_x(x)
        x_se = self.channel_adjust_x(x_se)  # 1x1 projection

        y_se = self.se_y(y)
        y_se = self.channel_adjust_y(y_se)  # project y channels -> x channels

        # weighted fusion before spatial attention
        z_0 = x_se + self.alpha * y_se

        # spatial relation modeling
        z_1 = self.rf(z_0)

        # concat original x (not x_se) with refined fusion for richer info
        fused = torch.cat([x, z_1], dim=1)  # channels = 2 * in_channels_x

        out = self.fusion_conv(fused)  # reduce back to in_channels_x

        # residual connection
        return out + x


if __name__ == '__main__':
    # quick test
    x = torch.randn(4, 16, 128, 128)
    y = torch.randn(4, 32, 128, 128)
    model = SFF(in_channels_x=16, in_channels_y=32, dropout=0.0)
    out = model(x, y)
    print("x:", x.shape)
    print("y:", y.shape)
    print("out:", out.shape)  # expect (4, 16, 128, 128)
