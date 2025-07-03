from tensorflow.keras import models, layers
from tensorflow.keras.layers import Add, Multiply

# Channel Attention Block
def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]
    shared_layer_one = layers.Dense(channel // ratio, activation='relu', use_bias=False)
    shared_layer_two = layers.Dense(channel, activation='sigmoid', use_bias=False)

    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = layers.Add()([avg_pool, max_pool])
    cbam_feature = layers.Multiply()([input_feature, cbam_feature])

    return cbam_feature

# Spatial Attention Block
def spatial_attention(input_feature):
    # Average pooling and max pooling across spatial dimensions
    avg_pool = layers.GlobalAveragePooling2D(keepdims=True)(input_feature)
    max_pool = layers.GlobalMaxPooling2D(keepdims=True)(input_feature)

    # Concatenate along the channel axis
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])

    # Apply a 7x7 convolution to generate the attention map
    spatial_attention = layers.Conv2D(filters=1, kernel_size=7, activation='sigmoid', padding='same', use_bias=False)(concat)

    # Multiply the attention map with the input feature map
    return layers.Multiply()([input_feature, spatial_attention])


# Residual Block
def residual_block(x, filters):
    shortcut = x
    x = layers.Conv2D(filters, 3, padding='same', activation='relu', use_bias=False)(x)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    # If the number of filters does not match, use Conv2D to match
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same', use_bias=False)(shortcut)
    x = Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    return x

# Multi-Scale Feature Extraction
def multi_scale_conv(x, filters):
    conv1 = layers.Conv2D(filters, 3, padding='same', activation='relu', use_bias=False)(x)
    conv2 = layers.Conv2D(filters, 5, padding='same', activation='relu', use_bias=False)(x)
    conv3 = layers.Conv2D(filters, 7, padding='same', activation='relu', use_bias=False)(x)
    concatenated = layers.Concatenate()([conv1, conv2, conv3])  # 3*filters
    # Reduce back to 'filters' using 1x1 convolution
    reduced = layers.Conv2D(filters, 1, padding='same', activation='relu', use_bias=False)(concatenated)
    return reduced

# Updated Model with Attention, Residual Blocks, and Multi-Scale Convolutions with Dropout
def create_model(ensem=0, dropout_rate=0.2):  # Added dropout_rate parameter
    inp = layers.Input(shape=(256, 256, 1))

    # Downsample path with dropout
    conv1 = multi_scale_conv(inp, 16)
    conv1 = residual_block(conv1, 16)
    pool1 = layers.MaxPool2D(2)(conv1)
    pool1_dropout = layers.Dropout(dropout_rate)(pool1)  # Added dropout

    conv2 = multi_scale_conv(pool1_dropout, 32)
    conv2 = residual_block(conv2, 32)
    pool2 = layers.MaxPool2D(2)(conv2)
    pool2_dropout = layers.Dropout(dropout_rate)(pool2)  # Added dropout

    conv3 = multi_scale_conv(pool2_dropout, 64)
    conv3 = residual_block(conv3, 64)
    pool3 = layers.MaxPool2D(2)(conv3)
    pool3_dropout = layers.Dropout(dropout_rate)(pool3)  # Added dropout

    conv4 = multi_scale_conv(pool3_dropout, 128)
    conv4 = residual_block(conv4, 128)
    pool4 = layers.MaxPool2D(2)(conv4)
    pool4_dropout = layers.Dropout(dropout_rate)(pool4)  # Added dropout

    bottleneck = multi_scale_conv(pool4_dropout, 256)  # Added dropout here
    bottleneck = residual_block(bottleneck, 256)

    # Apply Channel and Spatial Attention
    bottleneck_dropout = layers.Dropout(dropout_rate)(bottleneck) # Apply dropout before attention
    bottleneck_attention = channel_attention(bottleneck_dropout)
    bottleneck_attention = spatial_attention(bottleneck_attention)

    # Upsample path (no dropout here, typically)
    up4 = layers.Conv2DTranspose(128, 3, strides=2, activation='relu', padding='same', use_bias=False)(bottleneck_attention)
    concat4 = layers.concatenate([up4, conv4])
    up4_conv = residual_block(concat4, 128)

    up3 = layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same', use_bias=False)(up4_conv)
    concat3 = layers.concatenate([up3, conv3])
    up3_conv = residual_block(concat3, 64)

    up2 = layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same', use_bias=False)(up3_conv)
    concat2 = layers.concatenate([up2, conv2])
    up2_conv = residual_block(concat2, 32)

    up1 = layers.Conv2DTranspose(16, 3, strides=2, activation='relu', padding='same', use_bias=False)(up2_conv)
    concat1 = layers.concatenate([up1, conv1])
    up1_conv = residual_block(concat1, 16)

    if ensem == 1:
        model = models.Model(inputs=inp, outputs=up1_conv)
    else:
        final_output = layers.Conv2D(8, 1, activation='sigmoid', padding='same')(up1_conv)
        model = models.Model(inputs=inp, outputs=final_output)

    return model
