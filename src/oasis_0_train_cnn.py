"""
STEP 0a: Train 3D DenseNet CNN on OASIS-1 (CN vs AD) with LRP relevance maps.
Volumes are downsampled 4x to (44,52,44) for practical CPU training speed.
Run with: ~/miniconda3/envs/xai4dem/bin/python src/oasis_0_train_cnn.py

Outputs per fold:
  train/<exp_name>/cv_<k>/
    ADvsCN_cv<k>.h5         <- best model checkpoint
    Loss_train.png, Acc_train.png, ROC.png
    test_set_scores_<k>.csv
  train/<exp_name>/test_set_metrics.csv   <- summary across all folds
"""

import os, random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import innvestigate
import scipy.ndimage
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
import oasis_util as u

# ─── Config ──────────────────────────────────────────────────────────────────
EXP_NAME     = datetime.now().strftime('%Y-%m-%d_%H%M') + '_OASIS1_HIGHRES'
N_SPLITS     = 2           # Only 2 folds to save time (we only need fold 1 for clustering anyway)
LEARNING_RATE= 1e-4
NUM_EPOCHS   = 15          # early stopping will kick in before this usually
BATCH_SIZE   = 4           # reduced batch size for larger images
SEED         = 42
DOWNSAMPLE   = 2           # downsample factor: (176,208,176) -> (88,104,88) - 8x more volume!
SAVEPATH_BASE= f'../train/{EXP_NAME}'
os.makedirs(SAVEPATH_BASE, exist_ok=True)

# Downsampled input shape to use for CNN
DS_SHAPE = tuple(d // DOWNSAMPLE for d in u.IMAGE_SHAPE) + (1,)
print(f'Downsampled input shape: {DS_SHAPE}')

# ─── Load master CSV ─────────────────────────────────────────────────────────
df = pd.read_csv(u.MASTER_CSV)
df = df[df['label'].isin(['CN', 'AD'])].dropna(subset=['nifti_path']).reset_index(drop=True)
df = df[df['nifti_path'].apply(os.path.exists)].reset_index(drop=True)
df['label_int'] = (df['label'] == 'AD').astype(int)   # CN=0, AD=1
print(f"Dataset: {len(df)} subjects  |  CN={sum(df['label']=='CN')}  AD={sum(df['label']=='AD')}")

filenames = df['nifti_path'].tolist()
labels    = df['label_int'].tolist()

# ─── Helpers ─────────────────────────────────────────────────────────────────
def load_batch(paths):
    """Load a list of NIfTI paths → downsampled numpy array (N, H, W, D, 1)."""
    imgs = []
    for p in paths:
        vol = u.read_nifti(p)                            # (176,208,176)
        vol_ds = scipy.ndimage.zoom(                     # downsample 4x
            vol,
            zoom=[1/DOWNSAMPLE]*3,
            order=1                                      # linear interp
        )
        imgs.append(vol_ds[..., np.newaxis])             # add channel dim
    return np.array(imgs, dtype=np.float32)

def batch_generator(paths, labs, batch_size, shuffle=True):
    """Simple generator yielding (X_batch, y_batch) as one-hot."""
    idx = list(range(len(paths)))
    while True:
        if shuffle:
            random.shuffle(idx)
        for start in range(0, len(idx), batch_size):
            b = idx[start:start+batch_size]
            X = load_batch([paths[i] for i in b])
            y = tf.keras.utils.to_categorical([labs[i] for i in b], num_classes=2)
            yield X, y

# ─── 5-Fold Cross-Validation ────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
metrics_all = []

for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(filenames, labels)):
    k = fold_idx + 1
    print(f"\n{'='*60}\n FOLD {k}/{N_SPLITS}\n{'='*60}")
    savepath = os.path.join(SAVEPATH_BASE, f'cv_{k}')
    os.makedirs(savepath, exist_ok=True)

    # Split train/val (90/10 of the train_val portion)
    split = int(len(train_val_idx) * 0.9)
    train_idx = train_val_idx[:split]
    val_idx   = train_val_idx[split:]

    train_paths = [filenames[i] for i in train_idx]
    train_labs  = [labels[i] for i in train_idx]
    val_paths   = [filenames[i] for i in val_idx]
    val_labs    = [labels[i] for i in val_idx]
    test_paths  = [filenames[i] for i in test_idx]
    test_labs   = [labels[i] for i in test_idx]

    steps_train = max(1, len(train_paths) // BATCH_SIZE)
    steps_val   = max(1, len(val_paths)   // BATCH_SIZE)

    gen_train = batch_generator(train_paths, train_labs, BATCH_SIZE, shuffle=True)
    gen_val   = batch_generator(val_paths,   val_labs,   BATCH_SIZE, shuffle=False)

    # Compute class weights to handle imbalance
    n_ad = sum(1 for l in train_labs if l == 1)
    n_cn = len(train_labs) - n_ad
    total = len(train_labs)
    class_weight = {0: total / (2 * n_cn), 1: total / (2 * n_ad)}
    print(f"  Train: {len(train_paths)}  Val: {len(val_paths)}  Test: {len(test_paths)}")
    print(f"  Class weights: {class_weight}")

    # Build model on downsampled shape
    model = u.DenseNet(DS_SHAPE, u.N_CLASSES)
    opt   = tf.keras.optimizers.legacy.Adam(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    if fold_idx == 0:
        model.summary()

    ckpt_path = os.path.join(savepath, f'ADvsCN_cv{k:02d}.h5')
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, min_delta=0.01, verbose=1),
        ModelCheckpoint(filepath=ckpt_path, monitor='val_accuracy',
                        save_best_only=True, verbose=0)
    ]

    history = model.fit(gen_train, epochs=NUM_EPOCHS, steps_per_epoch=steps_train,
                        validation_data=gen_val, validation_steps=steps_val,
                        callbacks=callbacks, class_weight=class_weight, verbose=1)

    # Plot training curves
    for metric, label in [('loss','Loss'), ('accuracy','Accuracy')]:
        plt.figure()
        plt.plot(history.history[metric], 'bo', label=f'Train {label}')
        plt.plot(history.history[f'val_{metric}'], 'b', label=f'Val {label}')
        plt.title(label); plt.legend()
        plt.savefig(os.path.join(savepath, f'{metric}_train.png'))
        plt.close()

    # ── Evaluate on test set ─────────────────────────────────────────────────
    best_model = load_model(ckpt_path)
    X_test = load_batch(test_paths)
    y_test = tf.keras.utils.to_categorical(test_labs, num_classes=2)
    preds  = best_model.predict(X_test, verbose=0)

    fpr, tpr, _ = roc_curve(test_labs, preds[:, 1])
    roc_auc     = auc(fpr, tpr)
    acc         = accuracy_score(test_labs, np.argmax(preds, axis=1)) * 100

    print(f"  AUC: {roc_auc:.3f}  |  Acc: {acc:.1f}%")

    plt.figure()
    plt.plot([0, 1], [0, 1], 'navy', linestyle='--')
    plt.plot(fpr, tpr, color='red', label=f'AD vs CN (AUC={roc_auc:.2f})')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title(f'ROC Fold {k}'); plt.legend()
    plt.savefig(os.path.join(savepath, 'ROC.png')); plt.close()

    # Save per-subject predictions
    df_pred = pd.DataFrame({
        'Filename':      test_paths,
        'True_Label':    [u.CLASS_NAMES[l] for l in test_labs],
        'Pred_Label':    [u.CLASS_NAMES[np.argmax(p)] for p in preds],
        'CN_Prob':       preds[:, 0],
        'AD_Prob':       preds[:, 1],
    })
    df_pred.to_csv(os.path.join(savepath, f'test_set_scores_{k}.csv'), index=False)
    metrics_all.append({'fold': k, 'AUC': roc_auc, 'Acc': acc})

    # ── Generate LRP/IG heatmaps for first test sample (visual check) ────────
    try:
        model_wo_sm = innvestigate.model_wo_softmax(best_model)
        analyzer = innvestigate.create_analyzer(
            u.Relevance_Method[0][0], model_wo_sm, **u.Relevance_Method[0][1])

        sample_img = X_test[0:1]
        for neuron in [0, 1]:
            rel_path = os.path.join(savepath, f'relevance/neuron_{neuron}')
            os.makedirs(rel_path, exist_ok=True)
            anal = analyzer.analyze(sample_img, neuron_selection=neuron)
            a = np.squeeze(anal)
            a = scipy.ndimage.gaussian_filter(a, sigma=0.8)
            amax = np.quantile(np.abs(a), 0.9999)
            if amax > 0:
                a = a / amax
            for slice_i in range(0, a.shape[1], 5):
                plt.figure(figsize=(6, 4))
                plt.imshow(np.squeeze(sample_img[0, :, slice_i, :, 0]), cmap='gray')
                plt.imshow(a[:, slice_i, :], cmap='jet',
                           alpha=(np.abs(a[:, slice_i, :]) > 0.1).astype(float) * 0.5,
                           vmin=-1, vmax=1)
                plt.axis('off')
                plt.savefig(os.path.join(rel_path, f'slice_{slice_i}.png'),
                            bbox_inches='tight', dpi=80)
                plt.close()
    except Exception as e:
        print(f'  [WARN] Heatmap generation failed for fold {k}: {e}')

# ─── Save cross-fold metrics ─────────────────────────────────────────────────
df_metrics = pd.DataFrame(metrics_all)
df_metrics.to_csv(os.path.join(SAVEPATH_BASE, 'test_set_metrics.csv'), index=False)
print(f"\n✅ Done! Mean AUC: {df_metrics['AUC'].mean():.3f} ± {df_metrics['AUC'].std():.3f}")
print(f"   Mean Acc: {df_metrics['Acc'].mean():.1f}% ± {df_metrics['Acc'].std():.1f}%")
print(f"   All results saved to: {SAVEPATH_BASE}")
