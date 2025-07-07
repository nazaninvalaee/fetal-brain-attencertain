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
def hybrid_loss(y_true, y_pred, class_weights, dice_weight, ce_weight): # Add num_classes as a parameter if not global
    # --- IMPORTANT: Ensure y_true has the correct shape for one-hot encoding ---
    # y_true comes in as (batch_size, 256, 256, 1)
    # We need to squeeze out the last '1' dimension before one_hot
    y_true_squeezed = tf.squeeze(y_true, axis=-1) # Shape will be (batch_size, 256, 256)

    # Cast y_true to integer type for one-hot encoding
    y_true_one_hot = tf.one_hot(tf.cast(y_true_squeezed, tf.int32), depth=NUM_CLASSES)
    # y_true_one_hot now has shape (batch_size, 256, 256, NUM_CLASSES) - CORRECT!

    # --- Dice Loss Calculation ---
    # Apply softmax to y_pred if it's raw logits (common for segmentation models)
    y_pred_softmax = tf.nn.softmax(y_pred, axis=-1)

    # Add a small epsilon for numerical stability to avoid division by zero
    epsilon = 1e-7

    # Calculate intersection and union for Dice coefficient
    intersection = tf.reduce_sum(y_true_one_hot * y_pred_softmax, axis=[1, 2])
    union = tf.reduce_sum(y_true_one_hot + y_pred_softmax, axis=[1, 2])

    # Compute Dice coefficient and Dice loss per class
    dice_coefficient = (2. * intersection + epsilon) / (union + epsilon)
    dice_loss_per_class = 1. - dice_coefficient

    # Apply class weights to Dice Loss (assuming class_weights is a 1D tensor [NUM_CLASSES])
    weighted_dice_loss = tf.reduce_mean(dice_loss_per_class * class_weights) # Reduce mean across batch and classes

    # --- Cross-Entropy Loss Calculation ---
    # Use from_logits=True if y_pred is raw logits (recommended for stability)
    ce_loss_per_pixel = tf.nn.softmax_cross_entropy_with_logits(
        labels=y_true_one_hot,
        logits=y_pred # Use raw logits here
    )

    # Apply class weights to Cross-Entropy Loss
    # This typically involves re-weighting based on the true class of each pixel
    # If class_weights is (NUM_CLASSES,), you might need to gather weights for each pixel:
    # weighted_ce_loss_per_pixel = ce_loss_per_pixel * tf.gather(class_weights, tf.cast(y_true_squeezed, tf.int32))
    # Then take the mean over all pixels
    # For simplicity, if class_weights is meant to be applied globally after summing:
    ce_loss = tf.reduce_mean(ce_loss_per_pixel) # Mean over batch and pixels

    # Combine losses
    total_loss = (dice_weight * weighted_dice_loss) + (ce_weight * ce_loss)

    return total_loss
