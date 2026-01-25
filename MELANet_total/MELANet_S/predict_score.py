import os
import torch
import numpy as np
import tqdm
import sys

from PIL import Image
from tabulate import tabulate
from utils.eval_functions import *

def evaluate(pred_path, gt_path, verbose=True):
    # result_path = 'results/res2net50'
    result_path = 'results'
    method = 'result'
    Thresholds = np.linspace(1, 0, 256)
    headers = [
        'meanDic', 'meanIoU', 'wFm', 'Sm', 'meanEm', 'mae',
        'maxEm', 'maxDic', 'maxIoU', 'meanSen', 'maxSen',
        'meanSpe', 'maxSpe'
    ]
    results = []

    datasets = ['CAMO', 'COD10K', 'NC4K']
    if verbose:
        print('#' * 20, 'Start Evaluation', '#' * 20)
        datasets = tqdm.tqdm(
            datasets,
            desc='Expr - ' + method,
            total=len(datasets),
            position=0,
            bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}'
        )

    for dataset in datasets:
        pred_root = os.path.join(pred_path, dataset)
        gt_root = os.path.join(gt_path, dataset, 'GT')

        # 只读取常见图像后缀的文件，过滤掉目录
        valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        preds = [f for f in os.listdir(pred_root)
                 if f.lower().endswith(valid_ext) and
                 os.path.isfile(os.path.join(pred_root, f))]
        gts = [f for f in os.listdir(gt_root)
               if f.lower().endswith(valid_ext) and
               os.path.isfile(os.path.join(gt_root, f))]

        preds.sort()
        gts.sort()

        n = len(preds)
        threshold_Fmeasure = np.zeros((n, len(Thresholds)))
        threshold_Emeasure = np.zeros((n, len(Thresholds)))
        threshold_IoU = np.zeros((n, len(Thresholds)))
        threshold_Sensitivity = np.zeros((n, len(Thresholds)))
        threshold_Specificity = np.zeros((n, len(Thresholds)))
        threshold_Dice = np.zeros((n, len(Thresholds)))

        Smeasure = np.zeros(n)
        wFmeasure = np.zeros(n)
        MAE = np.zeros(n)

        if verbose:
            samples = tqdm.tqdm(
                enumerate(zip(preds, gts)),
                desc=dataset + ' - Evaluation',
                total=n,
                position=1,
                leave=False,
                bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}'
            )
        else:
            samples = enumerate(zip(preds, gts))

        for i, (pred, gt) in samples:
            # 保证预测图和 GT 同名（去掉后缀后相同）
            assert os.path.splitext(pred)[0] == os.path.splitext(gt)[0], \
                f"Mismatch: {pred} vs {gt}"

            pred_mask = np.array(
                Image.open(os.path.join(pred_root, pred))
            )
            gt_mask = np.array(
                Image.open(os.path.join(gt_root, gt))
            )

            # 如果读出来是三通道或更多，取第 0 维
            if pred_mask.ndim == 3:
                pred_mask = pred_mask[:, :, 0]
            if gt_mask.ndim == 3:
                gt_mask = gt_mask[:, :, 0]

            assert pred_mask.shape == gt_mask.shape, \
                f"Size mismatch: {pred_mask.shape} vs {gt_mask.shape}"

            # 规范到 0/1 的浮点下
            gt_mask = gt_mask.astype(np.float64) / 255.0
            gt_mask = (gt_mask > 0.5).astype(np.float64)

            pred_mask = pred_mask.astype(np.float64) / 255.0

            Smeasure[i] = StructureMeasure(pred_mask, gt_mask)
            wFmeasure[i] = original_WFb(pred_mask, gt_mask)
            MAE[i] = np.mean(np.abs(gt_mask - pred_mask))

            threshold_E = np.zeros(len(Thresholds))
            threshold_F = np.zeros(len(Thresholds))
            threshold_Pr = np.zeros(len(Thresholds))
            threshold_Rec = np.zeros(len(Thresholds))
            threshold_Iou = np.zeros(len(Thresholds))
            threshold_Spe = np.zeros(len(Thresholds))
            threshold_Dic = np.zeros(len(Thresholds))

            for j, threshold in enumerate(Thresholds):
                threshold_Pr[j], threshold_Rec[j], threshold_Spe[j], \
                    threshold_Dic[j], threshold_F[j], threshold_Iou[j] = \
                    Fmeasure_calu(pred_mask, gt_mask, threshold)

                Bi_pred = np.zeros_like(pred_mask)
                Bi_pred[pred_mask >= threshold] = 1
                threshold_E[j] = EnhancedMeasure(Bi_pred, gt_mask)

            threshold_Emeasure[i, :] = threshold_E
            threshold_Fmeasure[i, :] = threshold_F
            threshold_Sensitivity[i, :] = threshold_Rec
            threshold_Specificity[i, :] = threshold_Spe
            threshold_Dice[i, :] = threshold_Dic
            threshold_IoU[i, :] = threshold_Iou

        # 计算各指标
        mae = np.mean(MAE)
        Sm = np.mean(Smeasure)
        wFm = np.mean(wFmeasure)

        column_E = np.mean(threshold_Emeasure, axis=0)
        meanEm = np.mean(column_E)
        maxEm = np.max(column_E)

        column_Sen = np.mean(threshold_Sensitivity, axis=0)
        meanSen = np.mean(column_Sen)
        maxSen = np.max(column_Sen)

        column_Spe = np.mean(threshold_Specificity, axis=0)
        meanSpe = np.mean(column_Spe)
        maxSpe = np.max(column_Spe)

        column_Dic = np.mean(threshold_Dice, axis=0)
        meanDic = np.mean(column_Dic)
        maxDic = np.max(column_Dic)

        column_IoU = np.mean(threshold_IoU, axis=0)
        meanIoU = np.mean(column_IoU)
        maxIoU = np.max(column_IoU)

        metrics = [
            meanDic, meanIoU, wFm, Sm, meanEm, mae,
            maxEm, maxDic, maxIoU, meanSen, maxSen, meanSpe, maxSpe
        ]
        results.append([dataset, *metrics])

        # 将结果写入 CSV
        os.makedirs(result_path, exist_ok=True)
        csv_file = os.path.join(result_path, f'result_{dataset}.csv')
        write_header = not os.path.isfile(csv_file)
        with open(csv_file, 'a') as f:
            if write_header:
                f.write(','.join(['method', *headers]) + '\n')
            out_str = method + ',' + ','.join(f'{m:.4f}' for m in metrics) + '\n'
            f.write(out_str)

    tab = tabulate(results, headers=['dataset', *headers], floatfmt=".3f")
    if verbose:
        print(tab)
        print('#' * 20, 'End Evaluation', '#' * 20)
    return tab


if __name__ == "__main__":
    dataset_path = 'D:/ych/New/y_3/MELANet_Res2Net/TestDataset/'
    # dataset_path_pre = 'result'
    dataset_path_pre = 'results'
    evaluate(dataset_path_pre, dataset_path)
