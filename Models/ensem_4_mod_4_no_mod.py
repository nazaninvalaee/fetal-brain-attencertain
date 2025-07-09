from tensorflow.keras import models, layers
from tensorflow.keras.layers import Add, Multiply
from Models import layer_4_mod, layer_4_no_mod 

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

def create_model(dropout_rate=0.2, num_classes=8): 
    model1 = layer_4_mod.create_model(ensem=1, dropout_rate=dropout_rate)
    model2 = layer_4_no_mod.create_model(ensem=1, dropout_rate=dropout_rate)

    inp = layers.Input(shape=(256, 256, 1))

    out1 = model1(inp)
    out2 = model2(inp)

    conc1 = layers.concatenate([out1, out2])

    conc1 = channel_attention(conc1)

    conv2_for_gradcam = layers.Conv2D(16, 3, activation='relu', padding='same', name='gradcam_target_conv')(conc1)
    conv2_output = layers.Conv2D(16, 3, activation='relu', padding='same')(conv2_for_gradcam) 

    outp1 = layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(conv2_output)

    model = models.Model(inputs=inp, outputs=outp1)

    return model
