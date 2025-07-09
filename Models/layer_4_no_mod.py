from tensorflow.keras import models, layers
from tensorflow.keras.layers import Add


# Residual Block
def residual_block(x, filters):
    """
    A basic residual block with two Conv2D layers and a skip connection.

    Args:
        x (tf.Tensor): Input tensor.
        filters (int): Number of filters for Conv2D layers.

    Returns:
        tf.Tensor: Output tensor after residual addition and ReLU activation.
    """
    shortcut = x

    x = layers.Conv2D(filters, 3, padding='same', activation='relu', use_bias=False)(x)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same', use_bias=False)(shortcut)

    x = Add()([shortcut, x])
    x = layers.Activation('relu')(x)

    return x


# Multi-Scale Feature Extraction
def multi_scale_conv(x, filters):
    """
    Multi-scale convolution block using 3x3, 5x5, and 7x7 kernels.

    Args:
        x (tf.Tensor): Input tensor.
        filters (int): Output filters after dimensionality reduction.

    Returns:
        tf.Tensor: Output tensor with multi-scale features.
    """
    conv1 = layers.Conv2D(filters, 3, padding='same', activation='relu', use_bias=False)(x)
    conv2 = layers.Conv2D(filters, 5, padding='same', activation='relu', use_bias=False)(x)
    conv3 = layers.Conv2D(filters, 7, padding='same', activation='relu', use_bias=False)(x)

    concatenated = layers.Concatenate()([conv1, conv2, conv3])
    reduced = layers.Conv2D(filters, 1, padding='same', activation='relu', use_bias=False)(concatenated)

    return reduced


# U-Net Style Model with Residual Blocks and Multi-Scale Convolutions
def create_model(ensem=0, dropout_rate=0.2):
    """
    Builds a segmentation model with multi-scale convolutions and residual blocks.

    Args:
        ensem (int): If 1, return feature map before final Conv2D; otherwise return final output.
        dropout_rate (float): Dropout rate applied after pooling layers.

    Returns:
        tf.keras.Model: Compiled model.
    """
    inp = layers.Input(shape=(256, 256, 1))

    # Encoder
    conv1 = residual_block(multi_scale_conv(inp, 16), 16)
    pool1 = layers.MaxPool2D(2)(conv1)
    pool1 = layers.Dropout(dropout_rate)(pool1)

    conv2 = residual_block(multi_scale_conv(pool1, 32), 32)
    pool2 = layers.MaxPool2D(2)(conv2)
    pool2 = layers.Dropout(dropout_rate)(pool2)

    conv3 = residual_block(multi_scale_conv(pool2, 64), 64)
    pool3 = layers.MaxPool2D(2)(conv3)
    pool3 = layers.Dropout(dropout_rate)(pool3)

    conv4 = residual_block(multi_scale_conv(pool3, 128), 128)
    pool4 = layers.MaxPool2D(2)(conv4)
    pool4 = layers.Dropout(dropout_rate)(pool4)

    # Bottleneck
    bottleneck = residual_block(multi_scale_conv(pool4, 256), 256)
    bottleneck = layers.Dropout(dropout_rate)(bottleneck)

    # Decoder
    up4 = layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu', use_bias=False)(bottleneck)
    up4 = residual_block(layers.Concatenate()([up4, conv4]), 128)

    up3 = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu', use_bias=False)(up4)
    up3 = residual_block(layers.Concatenate()([up3, conv3]), 64)

    up2 = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu', use_bias=False)(up3)
    up2 = residual_block(layers.Concatenate()([up2, conv2]), 32)

    up1 = layers.Conv2DTranspose(16, 3, strides=2, padding='same', activation='relu', use_bias=False)(up2)
    up1 = residual_block(layers.Concatenate()([up1, conv1]), 16)

    if ensem == 1:
        model = models.Model(inputs=inp, outputs=up1)
    else:
        final_output = layers.Conv2D(8, 1, activation='sigmoid', padding='same')(up1)
        model = models.Model(inputs=inp, outputs=final_output)

    return model
