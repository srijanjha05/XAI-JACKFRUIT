import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import scipy.ndimage
import nibabel as nib

def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv3D):
            return layer.name
    return None

def make_gradcam_heatmap(img_array, tab_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model([img_array, tab_array])
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2, 3))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    # Notice we typically apply ReLU to the heatmap to only keep features that have positive influence
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def apply_gradcam_to_subject(pat_id, model, mri_vol_ds, tab_features, target_resolution=(176, 208, 176)):
    """
    Computes Grad-CAM for a given Multi-Modal model.
    mri_vol_ds should be shaped (1, 44, 52, 44, 1)
    tab_features should be shaped (1, 5)
    """
    last_conv = get_last_conv_layer_name(model)
    heatmap = make_gradcam_heatmap(mri_vol_ds, tab_features, model, last_conv, pred_index=1) # 1 = Alzheimer's

    # Upsample heatmap from the CNN latent shape (e.g., 5x6x5) back to the target resolution
    zoom_factors = (
        target_resolution[0] / heatmap.shape[0],
        target_resolution[1] / heatmap.shape[1],
        target_resolution[2] / heatmap.shape[2]
    )
    
    heatmap_full = scipy.ndimage.zoom(heatmap, zoom=zoom_factors, order=3)
    return heatmap_full
