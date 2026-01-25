# import torch
# import torch.nn as nn
# from einops import rearrange
# import torch.nn.functional as F
# from torch.nn.parameter import Parameter
# from lib.Mona import *


# class RFA(nn.Module):
#     def __init__(self, dim, num_heads=4, bias=False):
#         super(RFA, self).__init__()
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

#         self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
#         self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
#         self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

#     def forward(self, x):
#         b, c, h, w = x.shape
#         qkv = self.qkv_dwconv(self.qkv(x))
#         q, k, v = qkv.chunk(3, dim=1)

#         q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)

#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#         attn = attn.softmax(dim=-1)

#         out = (attn @ v)
#         out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
#         out = self.project_out(out)
#         return out + x


# # class SELayer(nn.Module):
# #     def __init__(self, channel, reduction=16):
# #         super(SELayer, self).__init__()
# #         self.avg_pool = nn.AdaptiveAvgPool2d(1)
# #         self.fc = nn.Sequential(
# #             nn.Linear(channel, max(channel // reduction, 1), bias=False),
# #             nn.ReLU(inplace=True),
# #             nn.Linear(max(channel // reduction, 1), channel, bias=False),
# #             nn.Sigmoid()
# #         )

# #     def forward(self, x):
# #         b, c, _, _ = x.size()
# #         y = self.avg_pool(x).view(b, c)
# #         return x * self.fc(y).view(b, c, 1, 1).expand_as(x)

# class SCLayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(SCLayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)  # 优化: 添加max pooling
        
#         reduced_channel = max(channel // reduction, 8)  # 最小为8，避免太小
#         self.fc = nn.Sequential(
#             nn.Linear(channel, reduced_channel, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(reduced_channel, channel, bias=False),
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         # 结合avg和max信息
#         y_avg = self.avg_pool(x).view(b, c)
#         y_max = self.max_pool(x).view(b, c)
        
#         # 共享FC层
#         y = self.fc(y_avg) + self.fc(y_max)
#         y = self.sigmoid(y).view(b, c, 1, 1)
        
#         return x * y.expand_as(x)


# class ChannelGate(nn.Module):
#     def __init__(self, gate_channels, reduction_ratio=16):
#         super(ChannelGate, self).__init__()
#         self.gate_channels = gate_channels
#         self.mlp = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(gate_channels, gate_channels // reduction_ratio),
#             nn.ReLU(),
#             nn.Linear(gate_channels // reduction_ratio, gate_channels)
#         )

#     def forward(self, x):
#         avg_out = self.mlp(F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
#         max_out = self.mlp(F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
#         channel_att_sum = avg_out + max_out

#         scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
#         return x * scale


# class SpatialGate(nn.Module):
#     def __init__(self):
#         super(SpatialGate, self).__init__()
#         kernel_size = 7
#         self.spatial = nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

#     def forward(self, x):
#         x_compress = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
#         x_out = self.spatial(x_compress)
#         scale = torch.sigmoid(x_out)  # broadcasting
#         return x * scale


# class CBAM(nn.Module):
#     def __init__(self, gate_channels, reduction_ratio=16):
#         super(CBAM, self).__init__()
#         self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
#         self.SpatialGate = SpatialGate()

#     def forward(self, x):
#         x_out = self.ChannelGate(x)
#         x_out = self.SpatialGate(x_out)
#         return x_out


# class CFAM(nn.Module):
#     """
#     Enhanced SFF:
#       - channel_adjust_x/y: 1x1 conv + BN + ReLU (keep lightweight)
#       - SE on both branches (as before)
#       - learnable alpha to weight y contribution: z0 = x_se + alpha * y_se
#       - RFA for spatial modeling
#       - fusion_conv (3x3) on concat([x, z1]) to produce final feature of same channels as x
#       - residual connection: out = fusion_out + x
#     """

#     def __init__(self, in_channels_x, in_channels_y, dropout=0.0):
#         super(CFAM, self).__init__()


#         self.channel_adjust_y = nn.Sequential(
#             nn.Conv2d(in_channels_y, in_channels_x, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(in_channels_x),
#             nn.ReLU(inplace=True)
#         )
#         self.channel_adjust_x = nn.Sequential(
#             nn.Conv2d(in_channels_x, in_channels_x, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(in_channels_x),
#             nn.ReLU(inplace=True)
#         )

#         # SE blocks
#         self.se_x = SCLayer(in_channels_x)
#         self.se_y = SCLayer(in_channels_y)

#         # RFA for spatial modelling (keeps channel = in_channels_x)
#         self.rf = RFA(in_channels_x)

#         # fusion conv: concat(x, z1) -> reduce back to in_channels_x
#         self.mona = Mona(in_dim=in_channels_x)


#     def channel_shuffle(self, x, groups):
#         b, c, h, w = x.shape

#         x = x.reshape(b, groups, -1, h, w)
#         x = x.permute(0, 2, 1, 3, 4)

#         # flatten
#         x = x.reshape(b, -1, h, w)

#         return x

#     def forward(self, x, y):
#         """
#         x: tensor with shape (B, in_channels_x, H, W)
#         y: tensor with shape (B, in_channels_y, H, W)
#         returns: tensor with shape (B, in_channels_x, H, W)
#         """
#         # SE attention
#         x_se = self.se_x(x)
#         x_se = self.channel_adjust_x(x_se)  # 1x1 projection

#         y_se = self.se_y(y)
#         y_se = self.channel_adjust_y(y_se)  # project y channels -> x channels

#         # weighted fusion before spatial attention
#         z_0 = x_se + y_se
#         indentity = z_0

#         # spatial relation modeling
#         z_1 = self.rf(z_0)
#         out = self.mona(z_1)
#         out = out + indentity

#         # residual connection
#         return self.channel_shuffle(out, 2)


# if __name__ == '__main__':
#     # quick test
#     x = torch.randn(4, 16, 128, 128)
#     y = torch.randn(4, 32, 128, 128)
#     model = CFAM(in_channels_x=16, in_channels_y=32, dropout=0.0)
#     out = model(x, y)
#     print("x:", x.shape)
#     print("y:", y.shape)
#     print("out:", out.shape)  # 输出: torch.Size([4, 16, 128, 128])

import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from lib.Mona import *


class GCCA(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(GCCA, self).__init__()
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




class SCLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SCLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 优化: 添加max pooling
        
        reduced_channel = max(channel // reduction, 8)  # 最小为8，避免太小
        self.fc = nn.Sequential(
            nn.Linear(channel, reduced_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channel, channel, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        # 结合avg和max信息
        y_avg = self.avg_pool(x).view(b, c)
        y_max = self.max_pool(x).view(b, c)
        
        # 共享FC层
        y = self.fc(y_avg) + self.fc(y_max)
        y = self.sigmoid(y).view(b, c, 1, 1)
        
        return x * y.expand_as(x)







class CFAM(nn.Module):


    def __init__(self, in_channels_x, in_channels_y, dropout=0.0):
        super(CFAM, self).__init__()

        self.proj= nn.Sequential(nn.Conv2d(in_channels_x + in_channels_y, in_channels_x, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(in_channels_x),
                                  nn.ReLU(inplace=True))
        


        # SE blocks
        self.se_x = SCLayer(in_channels_x)
        self.se_y = SCLayer(in_channels_y)

        # RFA for spatial modelling (keeps channel = in_channels_x)
        self.gcca = GCCA(in_channels_x)

        # fusion conv: concat(x, z1) -> reduce back to in_channels_x
        self.amsf = AMSF(in_dim=in_channels_x)


    def channel_shuffle(self, x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x, y):
        """
        x: tensor with shape (B, in_channels_x, H, W)
        y: tensor with shape (B, in_channels_y, H, W)
        returns: tensor with shape (B, in_channels_x, H, W)
        """
     
        x_se = self.se_x(x)
        y_se = self.se_y(y)
    


        # weighted fusion before spatial attention
        z_0 = torch.cat([x_se, y_se], dim=1)
        z_0 = self.channel_shuffle(z_0, 2)
        z_0 = self.proj(z_0) + x
        indentity = z_0

        # spatial relation modeling
        z_1 = self.gcca(z_0)
        out = self.amsf(z_1)
        out = out + indentity

        # residual connection
        return out


if __name__ == '__main__':
    # quick test
    x = torch.randn(4, 16, 128, 128)
    y = torch.randn(4, 32, 128, 128)
    model = CFAM(in_channels_x=16, in_channels_y=32, dropout=0.0)
    out = model(x, y)
    print("x:", x.shape)
    print("y:", y.shape)
    print("out:", out.shape)  # 输出: torch.Size([4, 16, 128, 128])















