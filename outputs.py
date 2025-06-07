import numpy as np
import tensorflow as tf # Import TensorFlow for model inference
from Testing.evaluation_metrics import *
from skimage.segmentation import find_boundaries
from tqdm import tqdm

# --- IMPORT YOUR MODEL HERE ---
# Uncomment ONE of the following lines based on which model you want to evaluate:
# from Models.layer_4_mod import create_model as create_segmentation_model
# from Models.layer_4_no_mod import create_model as create_segmentation_model
from Models.ensem_4_mod_4_no_mod import create_model as create_segmentation_model
# ------------------------------


# Function for getting the class for each pixel (remains largely the same)
def get_count(output, actual=[]):
    req = np.zeros((256, 256), dtype=np.uint8)
    # Ensure output is reshaped correctly for a single prediction before processing
    # output shape after squeezing is (256, 256, 8) or (1, 256, 256, 8) from model prediction
    if output.ndim == 4: # If batch dimension is present (e.g., (1, 256, 256, 8))
        out_probs = output[0] # Take the first (and only) sample in the batch
    else: # If it's already (256, 256, 8)
        out_probs = output

    # Argmax across the class dimension to get the predicted class for each pixel
    req = np.argmax(out_probs, axis=-1).astype(np.uint8)

    tp = np.zeros(8)
    fp = np.zeros(8)
    fn = np.zeros(8)

    if list(actual): # Only calculate TP/FP/FN if actual (ground truth) is provided
        for i in range(256):
            for j in range(256):
                predicted_class = req[i, j]
                true_class = actual[i, j]

                if predicted_class == true_class:
                    tp[predicted_class] += 1
                else:
                    fp[predicted_class] += 1
                    fn[true_class] += 1

    if not list(actual):
        return req

    return tp, fp, fn, req

# Function for calculating standard evaluation metrics for each class (remains the same)
def calc_met(tp, fp, fn, total, k):
    prec, dice, jac, sens = [], [], [], []

    for i in range(8):
        prec.append(precision(tp[i], fp[i]))
        sens.append(sensitivity(tp[i], fn[i]))
        dice.append(dice_score(tp[i], fp[i], fn[i]))
        jac.append(jaccard(tp[i], fp[i], fn[i]))

    if k == 1: # Return individual class metrics
        return prec, sens, jac, dice

    # Calculate averages if k == 0
    # Filter out 1s from metrics lists if class is not present (denominator was 0)
    # Or handle them as is, assuming 1 is desired for empty classes. Your current logic uses 1.
    avg_prec = round(np.mean(prec), 2)
    avg_dice = round(np.mean(dice), 2)
    avg_jac = round(np.mean(jac), 2)
    avg_sens = round(np.mean(sens), 2)
    acc = round(accuracy(sum(tp), total), 2)

    if k == 0: # Return averaged metrics
        return avg_prec, avg_sens, avg_jac, avg_dice, acc

# New Function: Calculate Boundary Precision and Recall (remains the same)
def calc_boundary_metrics(pred, actual):
    pred_boundary = find_boundaries(pred, mode='outer')
    actual_boundary = find_boundaries(actual, mode='outer')

    tp = np.sum(np.logical_and(pred_boundary, actual_boundary))
    fp = np.sum(np.logical_and(pred_boundary, np.logical_not(actual_boundary)))
    fn = np.sum(np.logical_and(np.logical_not(pred_boundary), actual_boundary))

    boundary_prec = precision(tp, fp)
    boundary_rec = sensitivity(tp, fn)

    return boundary_prec, boundary_rec

# Function for calculating and printing average and SD of metrics across all classes (remains the same)
def cal_avg_metric(metrics):
    metric_names = ["Precision", "Sensitivity", "Jaccard", "Dice Score", "Accuracy"]
    if len(metrics) == 2: # For boundary metrics
        metric_names = ["Boundary Precision", "Boundary Recall"]

    for c, metric in enumerate(metrics):
        mean_value = round(np.mean(metric), 2)
        std_value = round(np.std(metric), 2)
        if c < len(metric_names):
            print(f"\n{metric_names[c]} for the average of all brain parts:")
        else:
            print(f"\nMetrics no. {c+1} for the average of all brain parts:")
        print(f"Mean: {mean_value}, Std: {std_value}")

# Function for calculating and printing average and SD for each metric for each class (remains the same)
def cal_all_metric(metrics, num):
    metric_names = ["Precision", "Sensitivity", "Jaccard", "Dice Score"]
    if len(metrics) == 2: # For boundary metrics
        metric_names = ["Boundary Precision", "Boundary Recall"]

    for c, metric in enumerate(metrics):
        if c < len(metric_names):
            print(f"\n{metric_names[c]} for all brain parts:")
        else:
            print(f"\nMetrics no. {c+1} for all brain parts:")
        
        for i in range(8): # Iterate through 8 classes
            # Filter out lists that might contain single values if `all==0` was used for calc_met for some reason
            class_metric_values = [m[i] for m in metric if isinstance(m, list) and len(m) > i] # Ensure 'm' is a list and has enough elements
            if not class_metric_values: # Handle cases where a class might not have been present in any slice
                print(f"Class {i}: No data")
                continue
            mean_value = round(np.mean(class_metric_values), 2)
            std_value = round(np.std(class_metric_values), 2)
            print(f"Class {i}: Mean={mean_value}, Std={std_value}")


# Main function for prediction, evaluation, and boundary metric calculation with Uncertainty
def pred_and_eval_with_uncertainty(model, X_test, y_test=[], num_mc_passes=50, dropout_rate=0.2, all=0):
    """
    Performs segmentation prediction using MC Dropout and calculates evaluation metrics
    along with uncertainty maps (variance and entropy).

    Args:
        model: Your Keras model (already loaded or created by create_segmentation_model).
        X_test: Input MRI images for testing (numpy array).
        y_test: Ground truth segmentations (numpy array, optional).
        num_mc_passes: Number of forward passes for MC Dropout.
        dropout_rate: The dropout rate used in the model (for context/consistency, though not directly used in inference loop).
        all: If 0, returns average metrics; if 1, returns metrics per class.

    Returns:
        A dictionary containing:
        - 'segmentations': List of predicted segmentation masks.
        - 'precisions', 'sensitivities', 'jaccards', 'dice_scores': Lists of metric values.
        - 'boundary_precisions', 'boundary_recalls': Lists of boundary metric values.
        - 'variances': List of per-voxel, per-class variance maps.
        - 'entropies': List of per-voxel entropy maps.
        - 'accuracies': List of accuracies (if 'all' is 0 and y_test is provided).
    """

    # If X_test is a single image, wrap it in a list/batch dimension
    if X_test.ndim == 3: # If shape is (H, W, C) for a single image, convert to (1, H, W, C)
        X_test = np.expand_dims(X_test, axis=0)
        if list(y_test):
            y_test = np.expand_dims(y_test, axis=0) # Make y_test also batch-like

    prec_list, dice_list, jac_list, sens_list, acc_list = [], [], [], [], []
    boundary_prec_list, boundary_rec_list = [], []
    variance_list, entropy_list = [], []
    segmentation_list = []

    num_samples = X_test.shape[0]

    for k in tqdm(range(num_samples), desc="Executing with Uncertainty", ncols=75):
        current_input = X_test[k:k+1] # Take one sample at a time (maintains batch dim)
        current_actual = y_test[k] if list(y_test) else None

        # --- MC Dropout Inference ---
        mc_predictions_for_sample = []
        for _ in range(num_mc_passes):
            # Ensure model is called with training=True to activate dropout
            # The model is expecting (batch, H, W, C)
            output_prob = model(current_input, training=True).numpy() # Raw probabilities from sigmoid
            mc_predictions_for_sample.append(output_prob)
        
        # Convert list of (1, H, W, 8) arrays to (num_passes, 1, H, W, 8)
        mc_predictions_for_sample = np.array(mc_predictions_for_sample)
        
        # Squeeze the batch dimension (1) so it becomes (num_passes, H, W, 8)
        mc_predictions_for_sample = np.squeeze(mc_predictions_for_sample, axis=1)

        # --- Calculate Average Prediction for Final Segmentation ---
        avg_probabilities = np.mean(mc_predictions_for_sample, axis=0) # (H, W, 8)
        # Get the final class prediction by argmax
        segmentation = np.argmax(avg_probabilities, axis=-1).astype(np.uint8)
        segmentation_list.append(segmentation)

        # --- Calculate Metrics if Ground Truth is Provided ---
        if current_actual is not None:
            tp, fp, fn, _ = get_count(np.expand_dims(segmentation, axis=0), current_actual) # Pass seg as (1,H,W) to get_count

            if all == 0:
                prec, sens, jac, dice, acc = calc_met(tp, fp, fn, 256 * 256, all)
                acc_list.append(acc)
            elif all == 1:
                prec, sens, jac, dice = calc_met(tp, fp, fn, 256 * 256, all)

            prec_list.append(prec)
            dice_list.append(dice)
            jac_list.append(jac)
            sens_list.append(sens)

            # Boundary metrics
            boundary_prec, boundary_rec = calc_boundary_metrics(segmentation, current_actual)
            boundary_prec_list.append(boundary_prec)
            boundary_rec_list.append(boundary_rec)

        # --- Calculate Uncertainty Measures ---
        # Variance: Per-voxel, per-class variance of predicted probabilities
        # Shape: (H, W, 8)
        variance = np.var(mc_predictions_for_sample, axis=0)
        variance_list.append(variance)

        # Entropy: Per-voxel entropy of the average predicted probability distribution
        # Shape: (H, W)
        # Avoid log(0) by adding a small epsilon
        entropy = -np.sum(avg_probabilities * np.log(avg_probabilities + 1e-10), axis=-1)
        entropy_list.append(entropy)


    # --- Print or Return Results ---
    results = {
        'segmentations': np.array(segmentation_list), # Convert list to array for consistency
        'precisions': prec_list,
        'sensitivities': sens_list,
        'jaccards': jac_list,
        'dice_scores': dice_list,
        'boundary_precisions': boundary_prec_list,
        'boundary_recalls': boundary_rec_list,
        'variances': np.array(variance_list),
        'entropies': np.array(entropy_list)
    }

    if list(y_test): # Only print metrics if ground truth was provided
        if num_samples == 1:
            print('\n--- Metrics for Single Test Sample ---')
            print(f'Precision: {prec_list[0]}')
            print(f'Sensitivity: {sens_list[0]}')
            print(f'Jaccard: {jac_list[0]}')
            print(f'Dice Score: {dice_list[0]}')
            if all == 0:
                print(f'Accuracy: {acc_list[0]}')
            print(f'Boundary Precision: {boundary_prec_list[0]}')
            print(f'Boundary Recall: {boundary_rec_list[0]}')

        elif all == 0: # Print average metrics across samples
            print('\n--- Average Metrics Across Test Samples ---')
            cal_avg_metric([prec_list, sens_list, jac_list, dice_list, acc_list])
            cal_avg_metric([boundary_prec_list, boundary_rec_list])

        elif all == 1: # Print metrics per class across samples
            print('\n--- Metrics Per Class Across Test Samples ---')
            cal_all_metric([prec_list, sens_list, jac_list, dice_list], num_samples)
            cal_all_metric([boundary_prec_list, boundary_rec_list], num_samples)

    return results

# You would typically call this function from your main training/testing script like:
# from Models.create_dataset import create_dataset # Assuming this is how you get your data
#
# # 1. Load your test data (X_test, y_test)
# # X_train, X_test, y_train, y_test = create_dataset(path_to_input_mri, path_to_output_mri, s=0.2)
#
# # 2. Create/Load your model (e.g., the ensemble model)
# model = create_segmentation_model(dropout_rate=0.2) # Make sure you chose the correct import above!
# # If you have pre-trained weights: model.load_weights('path/to/your/weights.h5')
#
# # 3. Run evaluation with uncertainty
# mc_results = pred_and_eval_with_uncertainty(model, X_test, y_test, num_mc_passes=50, dropout_rate=0.2, all=0)
#
# # Now mc_results contains your segmentations, metrics, and uncertainty maps
# # You can then pass mc_results['entropies'] and mc_results['variances'] to your visualization functions.
