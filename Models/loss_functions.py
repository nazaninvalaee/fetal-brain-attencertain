import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np # Used for defining class_weights as a numpy array

# --- Dice Loss Function ---
def dice_loss(y_true, y_pred, smooth=1e-7):
    """
    Dice loss for segmentation.
    Args:
        y_true (tf.Tensor): Ground truth masks, one-hot encoded. Shape (batch_size, H, W, num_classes).
        y_pred (tf.Tensor): Predicted probabilities, one-hot encoded. Shape (batch_size, H, W, num_classes).
        smooth (float): Smoothing factor to prevent division by zero.
    Returns:
        tf.Tensor: Dice loss value.
    """
    # Flatten spatial and batch dimensions, keep class dimension
    # K.flatten works element-wise, so it flattens the last dimension of each batch item.
    # We want to flatten H, W for each channel. Reshape to (batch_size * H * W, num_classes)
    # Then sum along axis 0 for total contribution per class.
    # A more robust way to flatten for Dice:
    y_true_f = K.batch_flatten(y_true) # Flattens to (batch*H*W, num_classes)
    y_pred_f = K.batch_flatten(y_pred) # Flattens to (batch*H*W, num_classes)

    # Calculate intersection and union for each class
    intersection = K.sum(y_true_f * y_pred_f, axis=-1) # Sum over classes (batch*H*W,)
    union = K.sum(y_true_f + y_pred_f, axis=-1)       # Sum over classes (batch*H*W,)

    # Compute dice score for each class and average them
    dice = (2. * intersection + smooth) / (union + smooth)
    return K.mean(1 - dice) # Mean across the batch and flattened spatial elements

# --- Weighted Categorical Cross-Entropy Loss ---
def weighted_categorical_crossentropy(y_true, y_pred, class_weights):
    """
    Weighted Categorical Cross-Entropy Loss.
    Args:
        y_true (tf.Tensor): Ground truth masks, one-hot encoded. Shape (batch_size, H, W, num_classes).
        y_pred (tf.Tensor): Predicted probabilities. Shape (batch_size, H, W, num_classes).
        class_weights (tf.Tensor or np.array): A 1D tensor/array of shape (num_classes,)
                                               containing weights for each class.
    Returns:
        tf.Tensor: Weighted categorical cross-entropy loss value.
    """
    # Ensure y_pred is not exactly 0 or 1 to avoid log(0) issues
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

    # Calculate standard categorical cross-entropy per element
    crossentropy = -y_true * K.log(y_pred)

    # Apply class weights
    # class_weights should be (num_classes,)
    # Reshape class_weights to (1, 1, 1, num_classes) for broadcasting
    class_weights_tensor = K.constant(class_weights, dtype=tf.float32)
    class_weights_reshaped = K.reshape(class_weights_tensor, (1, 1, 1, -1))

    weighted_crossentropy = class_weights_reshaped * crossentropy

    # Sum over class dimension, then take mean over spatial and batch dimensions
    # to get a single scalar loss per batch
    return K.mean(K.sum(weighted_crossentropy, axis=-1)) # Sum over classes, then mean over spatial/batch

# --- Hybrid Loss Function (Dice + Weighted Categorical Cross-Entropy) ---
def hybrid_loss(y_true, y_pred, class_weights, dice_weight=0.5, ce_weight=0.5):
    """
    Combines Dice Loss and Weighted Categorical Cross-Entropy Loss.
    Args:
        y_true (tf.Tensor): Ground truth masks, typically integer encoded (batch_size, H, W).
                            Will be converted to one-hot inside.
        y_pred (tf.Tensor): Predicted probabilities. Shape (batch_size, H, W, num_classes).
        class_weights (tf.Tensor or np.array): Weights for each class in CE loss.
        dice_weight (float): Weight for the Dice loss component.
        ce_weight (float): Weight for the Weighted Categorical Cross-Entropy component.
    Returns:
        tf.Tensor: Combined loss value.
    """
    # Determine num_classes from y_pred's last dimension
    num_classes = K.int_shape(y_pred)[-1]

    # Convert y_true to one-hot encoding for both loss functions
    # Assuming y_true is (batch_size, H, W) with integer class IDs
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)

    # Calculate individual losses
    dice = dice_loss(y_true_one_hot, y_pred)
    ce = weighted_categorical_crossentropy(y_true_one_hot, y_pred, class_weights)

    # Combine them
    return (dice_weight * dice) + (ce_weight * ce)
