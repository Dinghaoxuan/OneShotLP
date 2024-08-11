# from model.model import OneShotLP
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from model.model_monkey import OneShotLP
# from model.model_qwen import OneShotLP
import torch
import cv2
from PIL import Image

import numpy as np
import cv2
from tqdm import tqdm

from utils import get_video_list, get_frame_list, path2image_np, img2video, frames2video, img_with_mask, mask2bbox

data_root = "/media/disk1/dhx/Dataset/UFPR-ALPR/testing/"

# demo_root = "/media/disk1/dhx/Project/LP_SAM/SAM_PT/video_demo"
# if not os.path.exists(demo_root):
#     os.makedirs(demo_root)

def remove_small_regions(mask: np.ndarray, area_thresh: float, mode: str):
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    """
    import cv2  # type: ignore

    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True


def ap(tp, conf, count):
    tp = np.array(tp)
    conf = np.array(conf)
    i = np.argsort(-conf)
    tp, conf = tp[i], conf[i]
    n_gt = count
    fpc = (1-tp[i]).cumsum()
    tpc = (tp[i]).cumsum()
    recall_curve = tpc / (n_gt + 1e-16)
    precision_curve = tpc / (tpc + fpc)

    ap = compute_ap(precision_curve, recall_curve)
    return ap

def compute_ap(precision, recall):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def iou(a,b):

    left1,top1,right1,down1 = a[0], a[1], a[2], a[3]
    left2,top2,right2,down2 = b[0], b[1], b[2], b[3]
    
    area1 = (right1-left1)*(top1-down1)
    area2 = (right2-left2)*(top2-down2)
    area_sum = area1+area2
    
    left = max(left1,left2)
    right = min(right1,right2)
    top = max(top1,top2)
    bottom = min(down1,down2)

    if left>=right or top>=bottom:
        return 0
    else:
        inter = (right-left)*(top-bottom)
        return inter/(area_sum-inter)

TP = 0
FP = 0
FN = 0
tp_list = []
conf_list = []
gt_count = 0
pred_count = 0

num_correct = 0
num_6correct = 0
num_GT = 0

LPD = OneShotLP(5, "crosshairs", None)

def calculate_accuracy(predicted, ground_truth):
    correct_characters = sum(p == g for p, g in zip(predicted, ground_truth))

    if correct_characters >= 6:
        return True
    else:
        return False
    

if __name__=="__main__":
    video_list = get_video_list(data_root)

    for idx, v in tqdm(enumerate(video_list)):
        
        torch.cuda.empty_cache()
        print(v)
        frames = get_frame_list(v)
        video, querys, annots, plates = frames2video(frames)


        video = video.cuda()

        query = annots[0] 

        with torch.no_grad():
            masks, masks_iou, pred_tracks, pred_plate = LPD(video, query, ref_neg_query=False)

        num_GT += len(plates)
        print('gt: ', plates[0])

        for i in range(len(plates)):
            try:
                for plate in pred_plate:
                    if calculate_accuracy(plate, plates[i]):
                        num_6correct += 1
                        break

                if plates[i] in pred_plate:
                    num_correct += 1
            except:
                continue
        
        gt = annots[1:]
        
        gt_count += len(gt)
        for t in range(len(masks)):
            mask, changed = remove_small_regions(masks[t], 5000, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, 5000, mode="islands")

            bboxs = mask2bbox(mask)
            pred_count += len(bboxs)
            _gt = gt[t]
            _gt[0,2] = _gt[0,2] + _gt[0,0]
            _gt[0,3] = _gt[0,3] + _gt[0,1]

            if len(bboxs) == 0:
                FN += 1
            else:
                for box in bboxs:
                    is_true = False
                    if iou(box, _gt[0]) >= 0.5:
                        is_true = True
                    if is_true:
                        TP += 1
                        tp_list.append(1.0)
                        conf_list.append(1.0)
                    else:
                        FP += 1
                        tp_list.append(0.0)
                        conf_list.append(0.0)

    P = TP / (pred_count + 1e-16)
    R = TP / (gt_count + 1e-16)
    F1 = 2 * P * R / (P + R + 1e-16)
    AP50 = ap(tp_list, conf_list, gt_count)
    ACC = num_correct / num_GT
    ACC6 = num_6correct / num_GT

    print('P: {:.4f}\t'.format(P),
      'R: {:.4f}\t'.format(R),
      'F1: {:.4f}\t'.format(F1),
      'AP50: {:.4f}\t'.format(AP50),
      'ACC@6: {:.4f}\t'.format(ACC6),
      'ACC@7: {:.4f}\t'.format(ACC))   



