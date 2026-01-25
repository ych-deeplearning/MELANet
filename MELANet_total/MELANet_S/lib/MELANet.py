import torch
import torch.nn.functional as F
import torch.nn as nn
from lib.modules import *
import torchvision.models as models
from torchvision.transforms.functional import rgb_to_grayscale
from einops import rearrange
from lib.Swim_encode import SwinTransformer
from lib.CFAM import *
# from lib.Swimv2_encoder import SwinTransformerV2
class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            Conv(in_channels, in_channels // 4), 
            Conv(in_channels // 4, out_channels)

        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        out = self.conv(x)
        return self.up(out)

class Out(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Out, self).__init__()
        self.conv1 = Conv(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x



#Multi-Scale Edge-Guided Attention Network
class MELANet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(MELANet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        # Encoder--Swimtransformer--
        self.encoder = SwinTransformer(img_size=384,
                                       embed_dim=128,  # 128, 192
                                       depths=[2, 2, 18, 2],
                                       num_heads=[4, 8, 16, 32],
                                       window_size=12)

        pretrained_dict = torch.load('/home/dell/User_student/ych/MELANet/lib/swin_base_patch4_window12_384_22k.pth')["model"]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(pretrained_dict)
        self.GMP = GMP()
        self.x5_dem_1 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.x4_dem_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(1024, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.x3_dem_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(512, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.x2_dem_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x1_dem_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.up5 = nn.Sequential(
            Conv(512, 512),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        ) 
        self.up4 = Up(768, 256)
        self.up3 = Up(384, 128)
        self.up2 = Up(192, 64)
        self.up1 = Up(128, 64)

        self.efm0 = CFAM(64, 64)
        self.efm1 = CFAM(64, 128)
        self.efm2 = CFAM(128, 256)
        self.efm3 = CFAM(256, 512)
        self.efm4 = CFAM(512, 1024)

        self.ega1 = MELAModule(64)
        self.ega2 = MELAModule(64)
        self.ega3 = MELAModule(128)
        self.ega4 = MELAModule(256)
        self.ega5 = MELAModule(512)
        
        self.out5 = Out(512, n_classes)
        self.out4 = Out(256, n_classes)
        self.out3 = Out(128, n_classes)
        self.out2 = Out(64, n_classes)
        self.out1 = Out(64, n_classes)

    def forward(self, x):
        grayscale_img = rgb_to_grayscale(x)
        edge_feature = make_laplace_pyramid(grayscale_img, 5, 1)
        edge_feature = edge_feature[1]

        #Encoder
        e5, e4, e3, e2, e1 = self.encoder(x)

        
        e5_dem_1 = self.x5_dem_1(e5)
        e4_dem_1 = self.x4_dem_1(e4)
        e3_dem_1 = self.x3_dem_1(e3)
        e2_dem_1 = self.x2_dem_1(e2)
        e1_dem_1 = self.x1_dem_1(e1)
        #Decoder
        out6 = self.GMP(e5)
        e5_dem_1 = self.efm4(e5_dem_1, e5)
        e5_dem_1 = self.ega5(edge_feature, e5_dem_1, out6)
        d5 = self.up5(e5_dem_1)
        e4_dem_1 = self.efm3(e4_dem_1, d5)
        out5 = self.out5(d5)
        ega4 = self.ega4(edge_feature, e4_dem_1, out5)

        d4 = self.up4(d5, ega4)
        e3_dem_1 = self.efm2(e3_dem_1, d4)
        out4 = self.out4(d4)    
        ega3 = self.ega3(edge_feature, e3_dem_1, out4)

        d3 = self.up3(d4, ega3)
        e2_dem_1 = self.efm1(e2_dem_1, d3)
        out3 = self.out3(d3)
        ega2 = self.ega2(edge_feature, e2_dem_1, out3)

        d2 = self.up2(d3, ega2)
        e1_dem_1 = self.efm0(e1_dem_1, d2)
        out2 = self.out2(d2)  
        ega1 = self.ega1(edge_feature, e1_dem_1, out2)

        d1 = self.up1(d2, ega1)
        out1 = self.out1(d1)
        
        return out1, out2, out3, out4, out5, out6


class MELANetModel(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(MELANetModel, self).__init__()
        self.channel = n_channels
        self.num_classes = n_classes
        self.net = MELANet(self.channel, self.num_classes)

    def forward(self, images):
        out1, out2, out3, out4, out5, out6 = self.net(images)
        out2 = F.interpolate(out2, size=images.size()[2:], mode='bilinear', align_corners=True)
        out3 = F.interpolate(out3, size=images.size()[2:], mode='bilinear', align_corners=True)
        out4 = F.interpolate(out4, size=images.size()[2:], mode='bilinear', align_corners=True)
        out5 = F.interpolate(out5, size=images.size()[2:], mode='bilinear', align_corners=True)
        out6 = F.interpolate(out6, size=images.size()[2:], mode='bilinear', align_corners=True)
        return out6, out5, out4, out3, out2, out1

