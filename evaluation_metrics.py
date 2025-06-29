import numpy as np
from skimage.segmentation import find_boundaries

# Jaccard similarity for checking the performance of the model
def jaccard(tp, fp, fn):  # same as iou
    den = fp + tp + fn
    if den == 0:
        return 1
    else:
        return round(tp / den, 2)

# Dice Score for checking the performance of the model
def dice_score(tp, fp, fn):  # same as f1-score
    den = fp + 2 * tp + fn
    if den == 0:
        return 1
    else:
        return round(2 * tp / den, 2)

# Precision for checking the performance of the model
def precision(tp, fp):
    den = tp + fp
    if den == 0:
        return 1
    else:
        return round(tp / den, 2)

# Sensitivity measure for checking the performance of the model
def sensitivity(tp, fn):  # same as recall
    den = tp + fn
    if den == 0:
        return 1
    else:
        return round(tp / den, 2)

# Accuracy for checking the performance of the model
def accuracy(tp, total):
    total_tp = np.sum(tp)
    return round(total_tp / total, 2)

# Boundary Precision: How well the model detects edges compared to ground truth
def boundary_precision(pred_mask, true_mask):
    pred_boundary = find_boundaries(pred_mask, mode='outer')
    true_boundary = find_boundaries(true_mask, mode='outer')
    
    tp = np.sum(np.logical_and(pred_boundary, true_boundary))
    fp = np.sum(np.logical_and(pred_boundary, np.logical_not(true_boundary)))

    return precision(tp, fp)

# Boundary Recall: How well the model's predicted boundaries align with the true boundaries
def boundary_recall(pred_mask, true_mask):
    pred_boundary = find_boundaries(pred_mask, mode='outer')
    true_boundary = find_boundaries(true_mask, mode='outer')
    
    tp = np.sum(np.logical_and(pred_boundary, true_boundary))
    fn = np.sum(np.logical_and(np.logical_not(pred_boundary), true_boundary))

    return sensitivity(tp, fn)
