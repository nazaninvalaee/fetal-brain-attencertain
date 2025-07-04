import numpy as np
import tensorflow as tf # Import TensorFlow for model inference
from Testing.evaluation_metrics import * # Assuming these are correctly defined
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
    # output shape after squeezing is (256, 256, NUM_CLASSES) or (1, 256, 256, NUM_CLASSES) from model prediction
    if output.ndim == 4: # If batch dimension is present (e.g., (1, 256, 256, NUM_CLASSES))
        out_probs = output[0] # Take the first (and only) sample in the batch
    else: # If it's already (256, 256, NUM_CLASSES)
        out_probs = output

    # Argmax across the class dimension to get the predicted class for each pixel
    req = np.argmax(out_probs, axis=-1).astype(np.uint8)

    # Determine NUM_CLASSES dynamically from the model output, or use a constant
    NUM_CLASSES = out_probs.shape[-1] # This is robust!

    tp = np.zeros(NUM_CLASSES)
    fp = np.zeros(NUM_CLASSES)
    fn = np.zeros(NUM_CLASSES)

    if list(actual): # Only calculate TP/FP/FN if actual (ground truth) is provided
        for i in range(256):
            for j in range(256):
                predicted_class = req[i, j]
                true_class = actual[i, j]

                # Ensure predicted_class and true_class are within NUM_CLASSES bounds
                # This handles potential issues if a class outside 0-NUM_CLASSES-1 is predicted/present
                if 0 <= predicted_class < NUM_CLASSES and 0 <= true_class < NUM_CLASSES:
                    if predicted_class == true_class:
                        tp[predicted_class] += 1
                    else:
                        fp[predicted_class] += 1
                        fn[true_class] += 1
                elif predicted_class >= NUM_CLASSES:
                    # Increment FP for out-of-bound prediction (might indicate a problem)
                    # Or simply ignore if this is expected for some reason. For now, treat as FP.
                    fp[predicted_class % NUM_CLASSES] += 1 # Map to a valid index if possible, or handle error
                elif true_class >= NUM_CLASSES:
                    # Increment FN for out-of-bound true class (might indicate a problem with labels)
                    fn[true_class % NUM_CLASSES] += 1


    if not list(actual):
        return req

    return tp, fp, fn, req

# Function for calculating standard evaluation metrics for each class (remains the same in logic)
def calc_met(tp, fp, fn, total, k):
    # Determine NUM_CLASSES from the length of tp/fp/fn arrays
    NUM_CLASSES = len(tp)

    prec, dice, jac, sens = [], [], [], []

    for i in range(NUM_CLASSES): # Use NUM_CLASSES here
        prec.append(precision(tp[i], fp[i]))
        sens.append(sensitivity(tp[i], fn[i]))
        dice.append(dice_score(tp[i], fp[i], fn[i]))
        jac.append(jaccard(tp[i], fp[i], fn[i]))

    if k == 1: # Return individual class metrics
        return prec, sens, jac, dice

    # Calculate averages if k == 0
    # Filter out 1s from metrics lists if class is not present (denominator was 0)
    # Or handle them as is, assuming 1 is desired for empty classes. Your current logic uses 1.
    # Note: If some metrics are 1.0 due to 0/0 (class not present), averaging them will skew the overall mean.
    # Consider filtering out classes with no ground truth or no predictions if a more 'true' average is desired.
    # For now, keeping your existing averaging logic.
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

# Function for calculating and printing average and SD for each metric for each class (remains the same in logic)
def cal_all_metric(metrics, num):
    metric_names = ["Precision", "Sensitivity", "Jaccard", "Dice Score"]
    # Determine NUM_CLASSES from the structure of the first metric's first element
    # Assuming metrics[0] is a list of lists, and metrics[0][0] is a list of class values
    NUM_CLASSES = len(metrics[0][0]) if metrics and isinstance(metrics[0], list) and metrics[0] else 0

    if len(metrics) == 2: # For boundary metrics (these are not per-class)
        metric_names = ["Boundary Precision", "Boundary Recall"]
        # Boundary metrics don't typically report per-class, so this section might be redundant for them.
        # Keeping it for consistency with original structure, but it will iterate for 8 "classes" for boundary
        # even if only a single value for boundary metrics is available per image.
        # This part of the function could be refactored for clarity if boundary metrics are always aggregated.

    for c, metric in enumerate(metrics):
        if c < len(metric_names):
            print(f"\n{metric_names[c]} for all brain parts:")
        else:
            print(f"\nMetrics no. {c+1} for all brain parts:")

        # Loop through classes for per-class metrics, but only if NUM_CLASSES > 0
        if NUM_CLASSES > 0 and len(metric_names) != 2: # Exclude boundary metrics from this loop
            for i in range(NUM_CLASSES): # Iterate through determined NUM_CLASSES
                # Filter out lists that might contain single values if `all==0` was used for calc_met for some reason
                class_metric_values = [m[i] for m in metric if isinstance(m, list) and len(m) > i]
                if not class_metric_values:
                    print(f"Class {i}: No data")
                    continue
                mean_value = round(np.mean(class_metric_values), 2)
                std_value = round(np.std(class_metric_values), 2)
                print(f"Class {i}: Mean={mean_value}, Std={std_value}")
        else: # For boundary metrics, or if NUM_CLASSES is 0 unexpectedly
             # Assuming boundary metrics are single values per image, not per-class
            if len(metric_names) == 2: # It's boundary metrics
                # This loop essentially aggregates the single boundary metric values over images
                mean_value = round(np.mean(metric), 2)
                std_value = round(np.std(metric), 2)
                print(f"Mean={mean_value}, Std={std_value}")
            else:
                print("No class-specific data to display.")


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
            output_prob = model(current_input, training=True).numpy()
            mc_predictions_for_sample.append(output_prob)

        # Convert list of (1, H, W, NUM_CLASSES) arrays to (num_passes, 1, H, W, NUM_CLASSES)
        mc_predictions_for_sample = np.array(mc_predictions_for_sample)

        # Squeeze the batch dimension (1) so it becomes (num_passes, H, W, NUM_CLASSES)
        mc_predictions_for_sample = np.squeeze(mc_predictions_for_sample, axis=1)

        # --- Calculate Average Prediction for Final Segmentation ---
        avg_probabilities = np.mean(mc_predictions_for_sample, axis=0) # (H, W, NUM_CLASSES)
        # Get the final class prediction by argmax
        segmentation = np.argmax(avg_probabilities, axis=-1).astype(np.uint8)
        segmentation_list.append(segmentation)

        # --- Calculate Metrics if Ground Truth is Provided ---
        if current_actual is not None:
            # Pass seg as (1,H,W) to get_count for consistent input shape handling
            # get_count will handle argmax internally, but here we already have argmaxed segmentation
            # We need to pass the raw probabilities to get_count or ensure get_count expects argmaxed input
            # Given get_count's current logic, it expects probabilities (H,W,NUM_CLASSES) or (1,H,W,NUM_CLASSES)
            # So, we should pass avg_probabilities, NOT the argmaxed 'segmentation' directly.
            tp, fp, fn, _ = get_count(np.expand_dims(avg_probabilities, axis=0), current_actual)

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
        # Shape: (H, W, NUM_CLASSES)
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
            # Pass individual lists of metric values for cal_avg_metric
            cal_avg_metric([prec_list, sens_list, jac_list, dice_list, acc_list])
            cal_avg_metric([boundary_prec_list, boundary_rec_list])

        elif all == 1: # Print metrics per class across samples
            print('\n--- Metrics Per Class Across Test Samples ---')
            # Pass individual lists of metric values for cal_all_metric
            cal_all_metric([prec_list, sens_list, jac_list, dice_list], num_samples)
            cal_all_metric([boundary_prec_list, boundary_rec_list], num_samples)

    return results
