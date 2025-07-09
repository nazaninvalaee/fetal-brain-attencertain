from tensorflow.keras import models, layers
from tensorflow.keras.layers import Add, Multiply


def channel_attention(input_feature, ratio=8):
    """
    Channel Attention Module using global average and max pooling.

    Args:
        input_feature (tf.Tensor): Feature map of shape (H, W, C).
        ratio (int): Bottleneck reduction ratio.

    Returns:
        tf.Tensor: Output after applying channel attention.
    """
    channel = input_feature.shape[-1]

    shared_dense_1 = layers.Dense(channel // ratio, activation='relu', use_bias=False)
    shared_dense_2 = layers.Dense(channel, activation='sigmoid', use_bias=False)

    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_dense_2(shared_dense_1(avg_pool))

    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_dense_2(shared_dense_1(max_pool))

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Multiply()([input_feature, cbam_feature])

    return cbam_feature


def spatial_attention(input_feature):
    """
    Spatial Attention Module using average and max pooling along channels.

    Args:
        input_feature (tf.Tensor): Feature map of shape (H, W, C).

    Returns:
        tf.Tensor: Output after applying spatial attention.
    """
    avg_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(input_feature)
    max_pool = layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(input_feature)

    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    attention = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid', use_bias=False)(concat)

    return Multiply()([input_feature, attention])


def residual_block(x, filters):
    """
    Residual block with optional projection shortcut.

    Args:
        x (tf.Tensor): Input tensor.
        filters (int): Number of filters.

    Returns:
        tf.Tensor: Output tensor.
    """
    shortcut = x

    x = layers.Conv2D(filters, 3, padding='same', activation='relu', use_bias=False)(x)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same', use_bias=False)(shortcut)

    x = Add()([shortcut, x])
    x = layers.Activation('relu')(x)

    return x


def multi_scale_conv(x, filters):
    """
    Multi-scale convolution block with 3x3, 5x5, and 7x7 filters.

    Args:
        x (tf.Tensor): Input tensor.
        filters (int): Number of output filters (after reduction).

    Returns:
        tf.Tensor: Output tensor.
    """
    conv1 = layers.Conv2D(filters, 3, padding='same', activation='relu', use_bias=False)(x)
    conv2 = layers.Conv2D(filters, 5, padding='same', activation='relu', use_bias=False)(x)
    conv3 = layers.Conv2D(filters, 7, padding='same', activation='relu', use_bias=False)(x)

    concatenated = layers.Concatenate()([conv1, conv2, conv3])
    reduced = layers.Conv2D(filters, 1, padding='same', activation='relu', use_bias=False)(concatenated)

    return reduced


def create_model(ensem=0, dropout_rate=0.2):
    """
    Creates an encoder-decoder model with attention and residual connections.

    Args:
        ensem (int): If 1, output intermediate features instead of final predictions.
        dropout_rate (float): Dropout rate to use after each pooling layer.

    Returns:
        tf.keras.Model: Compiled model.
    """
    inp = layers.Input(shape=(256, 256, 1))

    # Encoder
    conv1 = residual_block(multi_scale_conv(inp, 16), 16)
    pool1 = layers.Dropout(dropout_rate)(layers.MaxPool2D(2)(conv1))

    conv2 = residual_block(multi_scale_conv(pool1, 32), 32)
    pool2 = layers.Dropout(dropout_rate)(layers.MaxPool2D(2)(conv2))

    conv3 = residual_block(multi_scale_conv(pool2, 64), 64)
    pool3 = layers.Dropout(dropout_rate)(layers.MaxPool2D(2)(conv3))

    conv4 = residual_block(multi_scale_conv(pool3, 128), 128)
    pool4 = layers.Dropout(dropout_rate)(layers.MaxPool2D(2)(conv4))

    # Bottleneck
    bottleneck = residual_block(multi_scale_conv(pool4, 256), 256)
    bottleneck = layers.Dropout(dropout_rate)(bottleneck)
    bottleneck = channel_attention(bottleneck)
    bottleneck = spatial_attention(bottleneck)

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
        final_output = layers.Conv2D(8, 1, padding='same', activation='sigmoid')(up1)
        model = models.Model(inputs=inp, outputs=final_output)

    return model
