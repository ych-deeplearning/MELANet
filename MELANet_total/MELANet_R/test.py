import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from lib.MELANet import MELANetModel
import imageio
import torch.nn as nn
from utils.dataloader import test_dataset
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--pth_path', type=str, default='checkpointsNet_epoch_best.pth')

for _data_name in ['CAMO']:
    data_path = '/home/dell/User_student/ych/TestDataset/{}'.format(_data_name)
    save_path = './{}/'.format(_data_name)
    opt = parser.parse_args()
    model = MELANetModel()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        predicts= model(image)
        res = predicts[5]
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        imageio.imwrite(save_path+name, ((res>.5)*255).astype(np.uint8))
# import torch
# import torch.nn.functional as F
# import numpy as np
# import os, argparse
# from lib.MELANet import MELANetModel
# import imageio
# from utils.dataloader import test_dataset
#
#
# def load_model(model, pth_path):
#     """正确加载模型权重，处理DataParallel和checkpoint格式"""
#     print(f"Loading model from {pth_path}...")
#     checkpoint = torch.load(pth_path, map_location='cpu')
#
#     # 处理完整checkpoint（训练时保存的含epoch、optimizer等）
#     if 'model_state_dict' in checkpoint:
#         state_dict = checkpoint['model_state_dict']
#         print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
#     else:
#         state_dict = checkpoint
#         print("Loaded model weights directly")
#
#     # 处理DataParallel的module前缀
#     if next(iter(state_dict.keys())).startswith('module.'):
#         state_dict = {k[7:]: v for k, v in state_dict.items()}
#
#     # 检查模型与权重的键匹配情况
#     model_keys = set(model.state_dict().keys())
#     state_keys = set(state_dict.keys())
#
#     missing = model_keys - state_keys
#     unexpected = state_keys - model_keys
#
#     if missing:
#         print(f"Warning: Missing keys in state_dict: {missing}")
#     if unexpected:
#         print(f"Warning: Unexpected keys in state_dict: {unexpected}")
#
#     # 加载模型
#     model.load_state_dict(state_dict, strict=False)
#     print("Model loaded successfully")
#     return model
#
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--testsize', type=int, default=384, help='测试图像尺寸')
#     parser.add_argument('--pth_path', type=str, required=True, help='模型权重路径')
#     parser.add_argument('--dataset', type=str, default='NC4K', help='测试数据集名称')
#     args = parser.parse_args()   #90,116
#
#     # 确保结果目录存在
#     save_path = f'resul/res2net_new/{args.dataset}/'
#     os.makedirs(save_path, exist_ok=True)
#
#     # 初始化模型
#     model = MELANetModel()
#
#     # 加载模型
#     model = load_model(model, args.pth_path)
#     model.cuda()
#     model.eval()
#
#     # 准备测试数据
#     data_path = f'./TestDataset/{args.dataset}/'
#     image_root = f'{data_path}/Imgs/'
#     gt_root = f'{data_path}/GT/'
#     test_loader = test_dataset(image_root, gt_root, args.testsize)
#
#     print(f"开始测试数据集 {args.dataset}，共{test_loader.size}张图像...")
#     for i in range(test_loader.size):
#         image, gt, name = test_loader.load_data()
#         gt = np.asarray(gt, np.float32)
#         gt /= (gt.max() + 1e-8)  # 归一化真实标签
#
#         # 前向传播
#         image = image.cuda()
#         with torch.no_grad():
#             predicts = model(image)
#             res = predicts[0]  # 使用out1作为最终输出
#
#             # 调整尺寸
#             res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
#             res = res.data.cpu().numpy().squeeze()
#             res = (res - res.min()) / (res.max() - res.min() + 1e-8)  # 归一化预测结果
#
#             # 保存结果
#             imageio.imwrite(os.path.join(save_path, name), ((res > 0.5) * 255).astype(np.uint8))
#
#         if (i + 1) % 10 == 0:
#             print(f"处理进度: {i + 1}/{test_loader.size}")
#
#     print(f"测试完成，结果保存至 {save_path}")
#
#
# if __name__ == '__main__':
#     main()