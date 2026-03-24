"""
oasis_util.py — Shared utilities for OASIS-1 adapted pipeline.
Replaces segmentation_util.py for the OASIS-1 dataset.

Key differences from original:
- Binary classification: CN vs AD (no FTD)
- Image shape: (176, 208, 176) instead of (193, 229, 193)
- FSL_SEG tissue labels: 0=BG, 1=CSF, 2=GM, 3=WM
- OASIS naming: OAS1_XXXX_MR1
"""

import os
import numpy as np
import nibabel as nib
from keras import layers
from keras.layers import Input, Conv3D, BatchNormalization, Dense
from keras.layers import AveragePooling3D, MaxPooling3D
from keras.models import Model
from keras.layers import ReLU, concatenate
import keras.backend as K

# ─── Image config ────────────────────────────────────────────────────────────
IMAGE_SHAPE   = (176, 208, 176)      # T88 MNI space shape for OASIS-1
INPUT_SHAPE   = IMAGE_SHAPE + (1,)   # add channel dim
N_CLASSES     = 2                    # CN=0, AD=1
CLASS_NAMES   = ['CN', 'AD']

# ─── Paths (relative to src/) ────────────────────────────────────────────────
NIFTI_DIR     = '../data/nifti_converted'
MASTER_CSV    = '../data/oasis1_master.csv'
DATA_DISC_DIR = '../data'            # disc1..disc8 are here

# ─── XAI Method ──────────────────────────────────────────────────────────────
# LRP (default)
Relevance_Method_LRP = [
    ("lrp.sequential_preset_a",
     {"disable_model_checks": True, "neuron_selection_mode": "index", "epsilon": 1e-10},
     "LRP-CMPalpha1beta0")
]

# Integrated Gradients — switch to this for the IG run
Relevance_Method_IG = [
    ("integrated_gradients", {}, "IG")
]

# Change this to switch between methods:
Relevance_Method = Relevance_Method_LRP   # Switched back to LRP for the high-res run!

# ─── FSL Segmentation labels (tissue level) ──────────────────────────────────
# OASIS-1 FSL_SEG has 3 tissue classes (not 100+ regions like FastSurfer)
# 0 = Background, 1 = CSF, 2 = Gray Matter, 3 = White Matter
FSL_LABELS = {
    'Background': 0,
    'CSF':        1,
    'GrayMatter': 2,
    'WhiteMatter': 3,
}


# ─── DenseNet 3D model ───────────────────────────────────────────────────────
def DenseNet(ip_shape, op_shape):
    """
    3D DenseNet for binary CN vs AD classification.
    Slightly smaller than original (2 dense blocks of 3 reps each).
    """
    def bn_rl_conv(x, filters, kernel=1, strides=1):
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv3D(filters, kernel, strides=strides, padding='same')(x)
        return x

    def dense_block(x, repetition=3):
        for _ in range(repetition):
            y = bn_rl_conv(x, filters=8)
            y = bn_rl_conv(y, filters=8, kernel=3)
            x = concatenate([y, x])
        return x

    def transition_layer(x):
        x = bn_rl_conv(x, K.int_shape(x)[-1] // 2)
        x = AveragePooling3D(2, strides=2, padding='same')(x)
        return x

    inp = Input(ip_shape)
    x = Conv3D(10, 7, strides=2, padding='same')(inp)
    x = MaxPooling3D(3, strides=2, padding='same')(x)

    for rep in [3, 3]:
        d = dense_block(x, rep)
        x = transition_layer(d)

    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(d)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(rate=0.4)(x)
    x = layers.Flatten()(x)
    output = Dense(op_shape, activation='softmax', kernel_regularizer='l1_l2')(x)

    return Model(inp, output)


# ─── NIfTI loading ───────────────────────────────────────────────────────────
def read_nifti(filepath, minmax_scale=True):
    """Load a NIfTI file and optionally min-max scale it."""
    img = nib.load(filepath)
    data = np.squeeze(img.get_fdata())   # remove trailing dim if (H,W,D,1)
    data = np.nan_to_num(data)
    if minmax_scale:
        mn, mx = data.min(), data.max()
        if mx > mn:
            data = (data - mn) / (mx - mn)
    return data


# ─── Activation extraction ───────────────────────────────────────────────────
def get_sample_activation(analyzer, nifti_data, neuron):
    """
    Run iNNvestigate analyzer on one sample.
    nifti_data: 3D numpy array (H, W, D)
    Returns: 3D activation map same shape as nifti_data
    """
    x = nifti_data[np.newaxis, ..., np.newaxis]   # (1, H, W, D, 1)
    a = analyzer.analyze(x, neuron_selection=neuron)
    return np.squeeze(a)                            # back to (H, W, D)


# ─── FSL segmentation path for a given subject ───────────────────────────────
def find_fsl_seg_file(subject_id, data_disc_dir=DATA_DISC_DIR):
    """
    Find the FSL_SEG .img file for a given OASIS subject.
    Searches across all disc folders.
    """
    for disc in [f'disc{i}' for i in range(1, 9)]:
        seg_dir = os.path.join(data_disc_dir, disc, subject_id, 'FSL_SEG')
        if os.path.isdir(seg_dir):
            for f in os.listdir(seg_dir):
                if f.endswith('_fseg.img'):
                    # Convert .img → .nii.gz path in nifti folder? No — just load the .img directly.
                    return os.path.join(seg_dir, f)
    return None


# ─── Misc helpers ────────────────────────────────────────────────────────────
def sign_sum(arr, sign='pos'):
    """Sum only positive or negative values."""
    if sign == 'pos':
        return float(np.sum(np.where(arr < 0, 0, arr)))
    else:
        return float(np.sum(np.where(arr > 0, 0, arr)))


def dir_check(path, non_empty_ok=False):
    """Create directory; raise if non-empty (unless non_empty_ok=True)."""
    os.makedirs(path, exist_ok=True)
    if not non_empty_ok and len(os.listdir(path)) > 0:
        raise SystemExit(f'Directory not empty: {path}')
