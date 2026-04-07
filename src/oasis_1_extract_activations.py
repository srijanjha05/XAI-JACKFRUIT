"""
STEP 1: Extract multi-XAI attribution maps (LRP + IG) per subject and
        compute novel spatial features per FSL tissue region.

Novel features over Singh et al. (2025):
  1. Spatial Moments: mean, std, skewness, kurtosis, entropy of relevance
     distribution within each tissue — captures focal vs. diffuse atrophy
  2. Hemisphere Asymmetry Index (HAI): left vs. right hemisphere LRP
     relevance asymmetry per tissue — XAI-derived biomarker
  3. Concentration Ratio: fraction of positive relevance in top-10% voxels
     — measures sparsity of neural attention within a region

Runs both LRP (LRP-CMPalpha1beta0) and IG extractors in sequence.

Outputs per method:
  act_seg/<exp>/cv_<fold>/<method>/SegmentationActivation_node<N>.xlsx
    Sheets: act_sum_pos, act_sum_neg, act_total,
            density_pos, density_neg, density_total,
            no_voxels,
            spatial_moments, hai, concentration_ratio
"""

import os, logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').disabled  = True
logging.getLogger('innvestigate').disabled = True

import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.stats
import nibabel as nib
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import innvestigate
from keras.models import load_model
import oasis_util as u

# ─── Helpers for novel spatial features ──────────────────────────────────────

def relevance_entropy(values, n_bins=50):
    """
    Shannon entropy of the relevance value distribution within a tissue region.
    High entropy = diffuse/uniform relevance; low entropy = focal hotspot.
    Uses histogram-based probability estimate.
    """
    if len(values) == 0:
        return 0.0
    counts, _ = np.histogram(values, bins=n_bins)
    probs = counts / (counts.sum() + 1e-12)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def concentration_ratio(arr_masked, top_pct=0.10):
    """
    Fraction of total absolute relevance concentrated in the top top_pct voxels.
    CR = sum(top 10% largest |R_i|) / sum(|R_i|)
    High CR: focal/concentrated attention; Low CR: diffuse attention.
    """
    flat = arr_masked[arr_masked != 0]
    if len(flat) == 0:
        return 0.0
    abs_vals   = np.abs(flat)
    total_abs  = abs_vals.sum()
    if total_abs == 0:
        return 0.0
    n_top = max(1, int(len(abs_vals) * top_pct))
    top_sum = np.partition(abs_vals, -n_top)[-n_top:].sum()
    return float(top_sum / total_abs)


def hemisphere_asymmetry_index(activation, seg_vol, label_code, midline_x=None):
    """
    Computes the Hemisphere Asymmetry Index (HAI) for a given tissue region.
    Splits brain at the sagittal midline (x = midline_x) in MNI T88 space.

    HAI = (R_pos - L_pos) / (R_pos + L_pos + ε)
    where R_pos/L_pos = sum of positive relevance in right/left hemisphere
    within the tissue mask.

    HAI > 0: right hemisphere dominance
    HAI < 0: left hemisphere dominance
    AD shows left-temporal dominance in early stages → negative HAI for GM
    """
    H = activation.shape[0]
    if midline_x is None:
        midline_x = H // 2

    tissue_mask = (seg_vol == label_code)
    left_mask   = tissue_mask.copy()
    left_mask[midline_x:, :, :] = False
    right_mask  = tissue_mask.copy()
    right_mask[:midline_x, :, :] = False

    L_pos = float(np.sum(np.where(activation * left_mask  < 0, 0, activation * left_mask)))
    R_pos = float(np.sum(np.where(activation * right_mask < 0, 0, activation * right_mask)))
    hai   = (R_pos - L_pos) / (R_pos + L_pos + 1e-12)
    return hai, L_pos, R_pos


def spatial_moments(activation, mask):
    """
    Compute higher-order statistics of relevance values within a mask.
    Returns: mean, std, skewness, kurtosis, entropy
    """
    vals = activation[mask].flatten()
    if len(vals) == 0:
        return 0., 0., 0., 0., 0.
    return (
        float(np.mean(vals)),
        float(np.std(vals)),
        float(scipy.stats.skew(vals)),
        float(scipy.stats.kurtosis(vals)),   # excess kurtosis
        float(relevance_entropy(vals))
    )


# ─── Config ───────────────────────────────────────────────────────────────────
def get_latest_exp():
    train_dir = '../train'
    exps = [d for d in os.listdir(train_dir)
            if 'OASIS1' in d and os.path.isdir(os.path.join(train_dir, d))]
    exps.sort(key=lambda x: os.path.getmtime(os.path.join(train_dir, x)))
    return exps[-1] if exps else 'REPLACE_ME'

EXP_NAME = get_latest_exp()
CV_FOLD  = 1
NODE     = 1    # 1 = AD output node (relevance = contribution to AD prediction)

MODEL_PATH = f'../train/{EXP_NAME}/cv_{CV_FOLD}/ADvsCN_cv{CV_FOLD:02d}.h5'

# ─── Run for both XAI methods ─────────────────────────────────────────────────
METHODS_TO_RUN = u.Relevance_Method_LRP + u.Relevance_Method_IG

for (method_name, method_kwargs, method_label) in METHODS_TO_RUN:

    print(f'\n{"="*60}')
    print(f'  XAI Method: {method_label}')
    print(f'{"="*60}')

    SAVE_DIR = f'../act_seg/{EXP_NAME}/cv_{CV_FOLD}/{method_label}'
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f'Loading model: {MODEL_PATH}')
    model = load_model(MODEL_PATH)
    model_wo_sm = innvestigate.model_wo_softmax(model)
    analyzer = innvestigate.create_analyzer(method_name, model_wo_sm, **method_kwargs)

    df_master = pd.read_csv(u.MASTER_CSV)
    df_master = df_master[df_master['label'].isin(['CN', 'AD'])].reset_index(drop=True)
    df_master = df_master[df_master['nifti_path'].apply(os.path.exists)].reset_index(drop=True)

    regions = list(u.FSL_LABELS.keys())   # Background, CSF, GrayMatter, WhiteMatter
    base_cols = ['SubjectID', 'label'] + regions

    # Standard activation dataframes
    df_act_pos   = pd.DataFrame(columns=base_cols)
    df_act_neg   = pd.DataFrame(columns=base_cols)
    df_act_total = pd.DataFrame(columns=base_cols)
    df_dens_pos  = pd.DataFrame(columns=base_cols)
    df_dens_neg  = pd.DataFrame(columns=base_cols)
    df_dens_tot  = pd.DataFrame(columns=base_cols)
    df_voxels    = pd.DataFrame(columns=base_cols)

    # Novel spatial feature dataframes
    moment_cols = ['SubjectID', 'label']
    for r in ['CSF', 'GrayMatter', 'WhiteMatter']:
        for stat in ['mean', 'std', 'skew', 'kurt', 'entropy']:
            moment_cols.append(f'{r}_{stat}')
    df_moments = pd.DataFrame(columns=moment_cols)

    hai_cols = ['SubjectID', 'label']
    for r in ['CSF', 'GrayMatter', 'WhiteMatter']:
        hai_cols += [f'{r}_HAI', f'{r}_L_pos', f'{r}_R_pos']
    df_hai = pd.DataFrame(columns=hai_cols)

    cr_cols = ['SubjectID', 'label'] + [f'{r}_CR' for r in ['CSF', 'GrayMatter', 'WhiteMatter']]
    df_cr = pd.DataFrame(columns=cr_cols)

    errors = []
    midline_x = u.IMAGE_SHAPE[0] // 2  # = 88

    print(f'Processing {len(df_master)} subjects | Node: {NODE} | Method: {method_label}')

    for i, row in df_master.iterrows():
        sid   = row['ID']
        label = row['label']
        print(f'  [{i+1}/{len(df_master)}] {sid} ({label})', end=' ... ')

        # 1. FSL segmentation
        seg_path = u.find_fsl_seg_file(sid)
        if seg_path is None:
            errors.append(f'{sid}: No FSL_SEG file found')
            print('SKIP (no seg)')
            continue
        try:
            seg_vol = np.squeeze(nib.load(seg_path).get_fdata()).astype(int)
        except Exception as e:
            errors.append(f'{sid}: Seg load error — {e}')
            print('SKIP (seg error)')
            continue

        # 2. Compute attribution map
        try:
            nifti_data = u.read_nifti(row['nifti_path'], minmax_scale=True)
            nifti_ds   = scipy.ndimage.zoom(nifti_data,
                             zoom=(44/176, 52/208, 44/176), order=1)
            activation_ds = u.get_sample_activation(analyzer, nifti_ds, NODE)
            # Upsample back to full resolution with cubic spline
            activation = scipy.ndimage.zoom(activation_ds,
                             zoom=(176/44, 208/52, 176/44), order=3)
        except Exception as e:
            errors.append(f'{sid}: Activation error — {e}')
            print('SKIP (activation error)')
            continue

        # 3. Per-region standard statistics
        row_pos, row_neg, row_tot = [sid, label], [sid, label], [sid, label]
        row_dp, row_dn, row_dt    = [sid, label], [sid, label], [sid, label]
        row_vox = [sid, label]

        for region, lcode in u.FSL_LABELS.items():
            mask   = (seg_vol == lcode)
            n_vox  = int(np.sum(mask))
            seg_act = activation * mask

            ap = u.sign_sum(seg_act, 'pos')
            an = u.sign_sum(seg_act, 'neg')
            at = ap + an

            row_pos.append(ap); row_neg.append(an); row_tot.append(at)
            row_vox.append(n_vox)
            row_dp.append(ap / n_vox if n_vox > 0 else 0)
            row_dn.append(an / n_vox if n_vox > 0 else 0)
            row_dt.append(at / n_vox if n_vox > 0 else 0)

        df_act_pos.loc[len(df_act_pos)]    = row_pos
        df_act_neg.loc[len(df_act_neg)]    = row_neg
        df_act_total.loc[len(df_act_total)]= row_tot
        df_dens_pos.loc[len(df_dens_pos)]  = row_dp
        df_dens_neg.loc[len(df_dens_neg)]  = row_dn
        df_dens_tot.loc[len(df_dens_tot)]  = row_dt
        df_voxels.loc[len(df_voxels)]      = row_vox

        # 4. Novel: Spatial Moments per tissue
        moment_row = [sid, label]
        for r in ['CSF', 'GrayMatter', 'WhiteMatter']:
            mask  = (seg_vol == u.FSL_LABELS[r])
            mn, st, sk, ku, ent = spatial_moments(activation, mask)
            moment_row += [mn, st, sk, ku, ent]
        df_moments.loc[len(df_moments)] = moment_row

        # 5. Novel: Hemisphere Asymmetry Index
        hai_row = [sid, label]
        for r in ['CSF', 'GrayMatter', 'WhiteMatter']:
            hai_val, l_pos, r_pos = hemisphere_asymmetry_index(
                activation, seg_vol, u.FSL_LABELS[r], midline_x)
            hai_row += [hai_val, l_pos, r_pos]
        df_hai.loc[len(df_hai)] = hai_row

        # 6. Novel: Concentration Ratio
        cr_row = [sid, label]
        for r in ['CSF', 'GrayMatter', 'WhiteMatter']:
            mask    = (seg_vol == u.FSL_LABELS[r])
            seg_act = activation * mask
            cr_val  = concentration_ratio(seg_act)
            cr_row.append(cr_val)
        df_cr.loc[len(df_cr)] = cr_row

        print('OK')

    # ─── Save to Excel ────────────────────────────────────────────────────────
    out_file = os.path.join(SAVE_DIR, f'SegmentationActivation_node{NODE}.xlsx')
    with pd.ExcelWriter(out_file) as writer:
        df_act_pos.to_excel(writer,   sheet_name='act_sum_pos',     index=False)
        df_act_neg.to_excel(writer,   sheet_name='act_sum_neg',     index=False)
        df_act_total.to_excel(writer, sheet_name='act_total',       index=False)
        df_dens_pos.to_excel(writer,  sheet_name='density_pos',     index=False)
        df_dens_neg.to_excel(writer,  sheet_name='density_neg',     index=False)
        df_dens_tot.to_excel(writer,  sheet_name='density_total',   index=False)
        df_voxels.to_excel(writer,    sheet_name='no_voxels',       index=False)
        df_moments.to_excel(writer,   sheet_name='spatial_moments', index=False)
        df_hai.to_excel(writer,       sheet_name='hai',             index=False)
        df_cr.to_excel(writer,        sheet_name='concentration_ratio', index=False)

    print(f'\n✅ Saved: {out_file}')
    if errors:
        print(f'⚠️  {len(errors)} errors:')
        for e in errors: print(f'   - {e}')
        with open(os.path.join(SAVE_DIR, f'errors_node{NODE}.txt'), 'w') as f:
            f.writelines(line + '\n' for line in errors)

    del model, model_wo_sm, analyzer
    import gc; gc.collect()
    tf.keras.backend.clear_session()
