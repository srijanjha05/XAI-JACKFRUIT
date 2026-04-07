"""
STEP 0: Train 3D Multi-Modal DenseNet on OASIS-1 (CN vs AD).
Volumes downsampled 4x to (44,52,44) for CPU training.

Key improvements over baseline:
  - Focal loss (γ=2) to address class imbalance
  - Balanced batch sampling (50/50 CN/AD per batch)
  - 3D augmentation: random LR flip + Gaussian noise
  - Test-time augmentation (TTA) via LR-flip averaging
  - Threshold optimization on validation set
  - Fixed MMSE parsing (was corrupted for ~201 subjects)
  - MMSE_valid flag as an additional tabular feature

Run with: python src/oasis_0_train_cnn.py
"""

import os, random, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import scipy.ndimage
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import load_model
import keras.backend as K
import oasis_util as u

# ─── Config ──────────────────────────────────────────────────────────────────
EXP_NAME     = datetime.now().strftime('%Y-%m-%d_%H%M') + '_OASIS1_HIGHRES'
N_SPLITS     = 5
LEARNING_RATE= 5e-5
NUM_EPOCHS   = 50
BATCH_SIZE   = 8
SEED         = 42
DOWNSAMPLE   = 4          # (176,208,176) → (44,52,44)
FOCAL_GAMMA  = 2.0
FOCAL_ALPHA  = 0.25
TTA_FLIPS    = True       # test-time augmentation via LR flip averaging
SAVEPATH_BASE= f'../train/{EXP_NAME}'
os.makedirs(SAVEPATH_BASE, exist_ok=True)

DS_SHAPE = tuple(d // DOWNSAMPLE for d in u.IMAGE_SHAPE) + (1,)
print(f'Downsampled input shape: {DS_SHAPE}')

# ─── Focal Loss ───────────────────────────────────────────────────────────────
def focal_loss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA):
    """
    Focal loss (Lin et al. 2017) — down-weights easy examples so training
    focuses on hard, misclassified AD cases.
    FL(p_t) = -α * (1 - p_t)^γ * log(p_t)
    """
    def focal_loss_fn(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred  = K.clip(y_pred, epsilon, 1. - epsilon)
        ce      = -y_true * K.log(y_pred)
        weight  = K.pow(1.0 - y_pred, gamma)
        fl      = alpha * weight * ce
        return K.mean(K.sum(fl, axis=-1))
    return focal_loss_fn


# ─── Load and clean master CSV ───────────────────────────────────────────────
df = pd.read_csv(u.MASTER_CSV)
df = df[df['label'].isin(['CN', 'AD'])].dropna(subset=['nifti_path']).reset_index(drop=True)
df = df[df['nifti_path'].apply(os.path.exists)].reset_index(drop=True)
df['label_int'] = (df['label'] == 'AD').astype(int)

# Fix tabular features with robust parsing
df['Sex_int'] = (df['Sex'] == 'M').astype(float)

# MMSE: many entries have "eTIV: 1234" due to parsing bug — extract first number
def safe_numeric(series, vmin=None, vmax=None):
    """Extract first numeric value from string, clip to valid range."""
    s = series.astype(str).str.extract(r'(-?\d+\.?\d*)', expand=False)
    s = pd.to_numeric(s, errors='coerce')
    if vmin is not None:
        s = s.where(s >= vmin, other=np.nan)
    if vmax is not None:
        s = s.where(s <= vmax, other=np.nan)
    return s

df['MMSE_clean'] = safe_numeric(df['MMSE'], vmin=0, vmax=30)
df['MMSE_valid'] = df['MMSE_clean'].notna().astype(float)  # 1 if valid, 0 if missing
df['MMSE_clean'].fillna(df['MMSE_clean'].median(), inplace=True)

df['Age']   = safe_numeric(df['Age'],  vmin=0, vmax=120)
df['eTIV']  = safe_numeric(df['eTIV'], vmin=800, vmax=3000)
df['nWBV']  = safe_numeric(df['nWBV'], vmin=0.5, vmax=1.0)

feature_cols = ['Age', 'eTIV', 'nWBV', 'Sex_int', 'MMSE_clean', 'MMSE_valid']
for c in feature_cols:
    df[c].fillna(df[c].median(), inplace=True)
    df[c] = (df[c] - df[c].mean()) / (df[c].std() + 1e-8)

print(f"Dataset: {len(df)} subjects  |  CN={sum(df['label']=='CN')}  AD={sum(df['label']=='AD')}")

filenames = df['nifti_path'].tolist()
labels    = df['label_int'].tolist()
tab_feats = df[feature_cols].values.astype(np.float32)

# ─── Augmentation ────────────────────────────────────────────────────────────
def augment_volume(vol):
    """
    3D augmentation applied during training only.
    LR flip is biologically valid (AD pathology is bilateral; small flip variance
    reduces spatial overfitting without altering tissue distributions).
    """
    if np.random.random() < 0.5:
        vol = vol[::-1, :, :]               # sagittal LR flip
    if np.random.random() < 0.3:
        sigma = np.random.uniform(0.01, 0.03)
        vol = np.clip(vol + np.random.normal(0, sigma, vol.shape), 0, 1)
    return vol

def load_volume(path, augment=False):
    vol = u.read_nifti(path)
    vol_ds = scipy.ndimage.zoom(
        vol, zoom=(DS_SHAPE[0]/vol.shape[0], DS_SHAPE[1]/vol.shape[1], DS_SHAPE[2]/vol.shape[2]),
        order=1)
    if augment:
        vol_ds = augment_volume(vol_ds)
    return vol_ds[..., np.newaxis]

def load_batch(paths, augment=False):
    return np.array([load_volume(p, augment=augment) for p in paths], dtype=np.float32)


# ─── Balanced batch generator (50/50 CN/AD) ──────────────────────────────────
def balanced_generator(paths, tabs, labs, batch_size, augment=True):
    """
    Each batch contains exactly batch_size//2 CN and batch_size//2 AD samples,
    sampled with replacement from each class. This directly counteracts the
    3.36:1 CN:AD imbalance in OASIS-1.
    """
    cn_idx = [i for i, l in enumerate(labs) if l == 0]
    ad_idx = [i for i, l in enumerate(labs) if l == 1]
    half   = batch_size // 2
    while True:
        b_cn = np.random.choice(cn_idx, half, replace=True)
        b_ad = np.random.choice(ad_idx, half, replace=True)
        b    = np.concatenate([b_cn, b_ad])
        np.random.shuffle(b)
        X_img = load_batch([paths[i] for i in b], augment=augment)
        X_tab = np.array([tabs[i]  for i in b])
        y     = tf.keras.utils.to_categorical([labs[i] for i in b], num_classes=2)
        yield [X_img, X_tab], y


# ─── 5-Fold Cross-Validation ─────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
metrics_all = []

for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(filenames, labels)):
    k = fold_idx + 1
    print(f"\n{'='*60}\n FOLD {k}/{N_SPLITS}\n{'='*60}")
    savepath = os.path.join(SAVEPATH_BASE, f'cv_{k}')
    os.makedirs(savepath, exist_ok=True)

    split = int(len(train_val_idx) * 0.9)
    train_idx = train_val_idx[:split]
    val_idx   = train_val_idx[split:]

    train_paths = [filenames[i] for i in train_idx]
    train_tabs  = [tab_feats[i] for i in train_idx]
    train_labs  = [labels[i]    for i in train_idx]
    val_paths   = [filenames[i] for i in val_idx]
    val_tabs    = [tab_feats[i] for i in val_idx]
    val_labs    = [labels[i]    for i in val_idx]
    test_paths  = [filenames[i] for i in test_idx]
    test_tabs   = [tab_feats[i] for i in test_idx]
    test_labs   = [labels[i]    for i in test_idx]

    print(f"  Train: {len(train_paths)}  Val: {len(val_paths)}  Test: {len(test_paths)}")
    print(f"  Train AD: {sum(train_labs)}  Train CN: {len(train_labs)-sum(train_labs)}")

    steps_train = max(1, len(train_paths) // BATCH_SIZE)
    steps_val   = max(1, len(val_paths)   // BATCH_SIZE)

    gen_train = balanced_generator(train_paths, train_tabs, train_labs, BATCH_SIZE, augment=True)
    gen_val   = balanced_generator(val_paths,   val_tabs,   val_labs,   BATCH_SIZE, augment=False)

    # Build model
    model = u.MultiModalDenseNet(DS_SHAPE, tab_feats.shape[1], u.N_CLASSES)
    opt   = tf.keras.optimizers.legacy.Adam(lr=LEARNING_RATE)
    model.compile(loss=focal_loss(), optimizer=opt, metrics=['accuracy'])
    if fold_idx == 0:
        model.summary()

    ckpt_path = os.path.join(savepath, f'ADvsCN_cv{k:02d}.h5')
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=12, min_delta=0.005, verbose=1, restore_best_weights=True),
        ModelCheckpoint(filepath=ckpt_path, monitor='val_accuracy', save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]

    history = model.fit(
        gen_train, epochs=NUM_EPOCHS, steps_per_epoch=steps_train,
        validation_data=gen_val, validation_steps=steps_val,
        callbacks=callbacks, verbose=1)

    # Training curves
    for metric, label in [('loss','Loss'), ('accuracy','Accuracy')]:
        plt.figure()
        plt.plot(history.history[metric],     'bo', label=f'Train {label}')
        plt.plot(history.history[f'val_{metric}'], 'b', label=f'Val {label}')
        plt.title(label); plt.legend()
        plt.savefig(os.path.join(savepath, f'{metric}_train.png'))
        plt.close()

    # ── Evaluate with Threshold Optimization ────────────────────────────────
    best_model = load_model(ckpt_path, custom_objects={'focal_loss_fn': focal_loss()})

    # Load validation set for threshold optimization
    X_val_img = load_batch(val_paths, augment=False)
    X_val_tab = np.array(val_tabs)
    val_probs = best_model.predict([X_val_img, X_val_tab], verbose=0)

    # Find threshold that maximizes F1 (balanced: no false economy on sensitivity)
    best_thresh = 0.5
    best_f1     = 0.0
    for t in np.arange(0.15, 0.85, 0.01):
        preds_t  = (val_probs[:, 1] >= t).astype(int)
        tp = sum(1 for p, g in zip(preds_t, val_labs) if p == 1 and g == 1)
        fp = sum(1 for p, g in zip(preds_t, val_labs) if p == 1 and g == 0)
        fn = sum(1 for p, g in zip(preds_t, val_labs) if p == 0 and g == 1)
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        if f1 > best_f1:
            best_f1     = f1
            best_thresh = t

    print(f"  Optimal threshold: {best_thresh:.2f}  (val F1={best_f1:.3f})")

    # ── Test set evaluation with TTA ─────────────────────────────────────────
    X_test_img  = load_batch(test_paths, augment=False)
    X_test_tab  = np.array(test_tabs)
    preds_orig  = best_model.predict([X_test_img, X_test_tab], verbose=0)

    if TTA_FLIPS:
        # Predict on LR-flipped versions and average
        X_test_flip = X_test_img[:, ::-1, :, :, :]  # flip x-axis
        preds_flip  = best_model.predict([X_test_flip, X_test_tab], verbose=0)
        preds       = (preds_orig + preds_flip) / 2.0
    else:
        preds = preds_orig

    fpr, tpr, _ = roc_curve(test_labs, preds[:, 1])
    roc_auc     = auc(fpr, tpr)

    # Accuracy with optimized threshold
    pred_labels = (preds[:, 1] >= best_thresh).astype(int)
    acc         = accuracy_score(test_labs, pred_labels) * 100

    cm = confusion_matrix(test_labs, pred_labels)
    print(f"  AUC: {roc_auc:.3f}  |  Acc: {acc:.1f}%  |  Threshold: {best_thresh:.2f}")
    print(f"  Confusion matrix (CN=0, AD=1):\n{cm}")

    plt.figure()
    plt.plot([0, 1], [0, 1], 'navy', linestyle='--')
    plt.plot(fpr, tpr, color='red', label=f'AD vs CN (AUC={roc_auc:.2f})')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title(f'ROC Fold {k}'); plt.legend()
    plt.savefig(os.path.join(savepath, 'ROC.png')); plt.close()

    # Save per-subject predictions
    df_pred = pd.DataFrame({
        'Filename':   test_paths,
        'True_Label': [u.CLASS_NAMES[l] for l in test_labs],
        'Pred_Label': [u.CLASS_NAMES[p] for p in pred_labels],
        'CN_Prob':    preds[:, 0],
        'AD_Prob':    preds[:, 1],
        'Threshold':  best_thresh,
    })
    df_pred.to_csv(os.path.join(savepath, f'test_set_scores_{k}.csv'), index=False)
    metrics_all.append({'fold': k, 'AUC': roc_auc, 'Acc': acc, 'Threshold': best_thresh})

    # Save threshold for downstream XAI scripts
    with open(os.path.join(savepath, 'threshold.json'), 'w') as f:
        json.dump({'threshold': float(best_thresh), 'val_f1': float(best_f1)}, f)

    # Only need fold 1 for downstream XAI pipeline
    if k == 1:
        break

# ─── Save summary ─────────────────────────────────────────────────────────────
df_metrics = pd.DataFrame(metrics_all)
df_metrics.to_csv(os.path.join(SAVEPATH_BASE, 'test_set_metrics.csv'), index=False)
print(f"\n✅ Done! Mean AUC: {df_metrics['AUC'].mean():.3f} ± {df_metrics['AUC'].std():.3f}")
print(f"   Mean Acc: {df_metrics['Acc'].mean():.1f}% ± {df_metrics['Acc'].std():.1f}%")
print(f"   Results: {SAVEPATH_BASE}")
