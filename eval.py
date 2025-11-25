# from utils.test_utils_res import SegFormer_Segmentation
import cv2
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm
import os
import torch
import warnings
warnings.filterwarnings("ignore")
torch.cuda.empty_cache()

def metric(premask, groundtruth):
    seg_inv, gt_inv = np.logical_not(premask), np.logical_not(groundtruth)
    true_pos = float(np.logical_and(premask, groundtruth).sum())  # float for division
    true_neg = np.logical_and(seg_inv, gt_inv).sum()
    false_pos = np.logical_and(premask, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, groundtruth).sum()
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    cross = np.logical_and(premask, groundtruth)
    union = np.logical_or(premask, groundtruth)
    iou = np.sum(cross) / (np.sum(union) + 1e-6)
    if np.sum(cross) + np.sum(union) == 0:
        iou = 1
    return f1, iou


def evaluate(path_pre,path_gt,dataset_name,record_txt):
    if os.path.exists(path_gt):
        flist = sorted(os.listdir(path_pre))
        auc, f1, iou = [], [], []
        for file in tqdm(flist):
            try:
                pre = cv2.imread(path_pre + file)

                # gt = cv2.imread(path_gt + file[:-4] + '.png') #C1

                # gt = cv2.imread(path_gt + file[:-4] + '.tif') #Coverage
                
                # gt = cv2.imread(path_gt + file) # NC16

                gt = cv2.imread(path_gt + file[:-4]+'.png') #Columbia

                H, W, C = pre.shape
                Hg, Wg, C = gt.shape
                if H != Hg or W != Wg:
                    print("!=")
                    # gt = cv2.resize(gt, (W, H))
                    # gt[gt > 127] = 255
                    # gt[gt <= 127] = 0
                if np.max(gt) != np.min(gt):
                    auc.append(roc_auc_score((gt.reshape(H * W * C) / 255).astype('int'), pre.reshape(H * W * C) / 255.))
                pre[pre > 127] = 255
                pre[pre <= 127] = 0
                a, b = metric(pre / 255, gt / 255)
                f1.append(a)
                iou.append(b)
            except Exception as e:
                print(file)

        print(dataset_name)
        print('Evaluation: AUC: %5.4f, F1: %5.4f, IOU: %5.4f' % (np.mean(auc), np.mean(f1), np.mean(iou)))
    with open(record_txt,"a") as f:
        f.writelines(dataset_name+"\n")
        f.writelines('Evaluation: AUC: %5.4f, F1: %5.4f, IOU: %5.4f' % (np.mean(auc), np.mean(f1), np.mean(iou)))
        f.writelines("\n")
    return np.mean(auc), np.mean(f1), np.mean(iou)

if __name__ == "__main__":
    save_path = '/home/lzy/lsl-project/LCNet-cf/results_cf/NC16_Train2/xiaorong/B_MEFM2_bs/Columbia/'
    path_gt = '/home/lzy/lsl-project/SCIML/SCWSIML/CodDataset/test/Columbia/test/GT/'
    record_txt = r"./test_out/zoom.txt"
    with open(record_txt,"a") as f:
        # f.writelines(str(used_weigth))
        f.writelines("\n")

    auc,f1,iou=evaluate(save_path,path_gt,"B_MEFM2_bs_Columbia",record_txt)