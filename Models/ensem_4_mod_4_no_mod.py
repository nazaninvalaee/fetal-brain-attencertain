from tensorflow.keras import models, layers
from tensorflow.keras.layers import Add, Multiply
from Models import layer_4_mod, layer_4_no_mod # Ensure these are correctly imported

# Channel Attention Block for the Ensemble
def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]
    shared_layer_one = layers.Dense(channel // ratio, activation='relu', use_bias=False)
    shared_layer_two = layers.Dense(channel, activation='sigmoid', use_bias=False) # This sigmoid is for attention weights, not final classification, so it's fine.

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

# Ensemble model - MODIFIED TO ACCEPT NUM_CLASSES
def create_model(dropout_rate=0.2, num_classes=8): # <--- CHANGED DEFAULT from 7 to 8
    # Load both models, passing the dropout_rate
    # IMPORTANT: Also ensure layer_4_mod.py and layer_4_no_mod.py
    # are NOT outputting a softmax/sigmoid themselves.
    # They should output raw features/logits if they are part of a larger ensemble's final classification.
    # If their create_model functions accept a final_activation parameter, it should be None or linear.
    model1 = layer_4_mod.create_model(ensem=1, dropout_rate=dropout_rate)
    model2 = layer_4_no_mod.create_model(ensem=1, dropout_rate=dropout_rate)

    # Input
    inp = layers.Input(shape=(256, 256, 1))

    # Get outputs from both models
    out1 = model1(inp)
    out2 = model2(inp)

    # Concatenate the outputs
    conc1 = layers.concatenate([out1, out2])

    # Attention on the combined output
    conc1 = channel_attention(conc1)
    
    # Further refinement with convolution layers
    # Give this layer a name so we can access it later for Grad-CAM
    conv2_for_gradcam = layers.Conv2D(16, 3, activation='relu', padding='same', name='gradcam_target_conv')(conc1)
    conv2_output = layers.Conv2D(16, 3, activation='relu', padding='same')(conv2_for_gradcam) # This layer takes input from the named layer

    # Final output layer
    outp1 = layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(conv2_output)

    # Create the final model
    model = models.Model(inputs=inp, outputs=outp1)

    return model
