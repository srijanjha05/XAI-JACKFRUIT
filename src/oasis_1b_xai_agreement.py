"""
STEP 1b: Multi-XAI Inter-Method Agreement Scoring (NOVEL)

For each subject, computes the per-tissue Spearman rank correlation between
LRP and IG attribution maps. This agreement score serves as a per-subject
confidence/reliability measure for the XAI explanation:

  - High agreement (rho ≈ 1): both methods consistently highlight the same
    voxels → robust, reliable attribution → high confidence explanation
  - Low agreement (rho ≈ 0): methods disagree on which voxels matter →
    uncertain/unstable attribution for this subject

This is a novel contribution not present in Singh et al. (2025), who use
only LRP. The agreement score enriches the explanation space by quantifying
per-subject XAI reliability, which is clinically meaningful:
subjects with low agreement may represent atypical presentations.

Outputs:
  act_seg/<exp>/cv_<fold>/xai_agreement_node<N>.csv
    Columns: SubjectID, label, CSF_agree, GrayMatter_agree, WhiteMatter_agree,
             CSF_agree_p, GrayMatter_agree_p, WhiteMatter_agree_p,
             mean_agreement
"""

import os, logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').disabled  = True
logging.getLogger('innvestigate').disabled = True

import numpy as np
import pandas as pd
import scipy.stats
import scipy.ndimage
import nibabel as nib
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import innvestigate
from keras.models import load_model
import oasis_util as u

def get_latest_exp():
    train_dir = '../train'
    exps = [d for d in os.listdir(train_dir)
            if 'OASIS1' in d and os.path.isdir(os.path.join(train_dir, d))]
    exps.sort(key=lambda x: os.path.getmtime(os.path.join(train_dir, x)))
    return exps[-1] if exps else 'REPLACE_ME'

EXP_NAME = get_latest_exp()
CV_FOLD  = 1
NODE     = 1

MODEL_PATH = f'../train/{EXP_NAME}/cv_{CV_FOLD}/ADvsCN_cv{CV_FOLD:02d}.h5'
SAVE_DIR   = f'../act_seg/{EXP_NAME}/cv_{CV_FOLD}'
os.makedirs(SAVE_DIR, exist_ok=True)

# ─── Load model once, create both analyzers ───────────────────────────────────
print(f'Loading model: {MODEL_PATH}')
model = load_model(MODEL_PATH)
model_wo_sm = innvestigate.model_wo_softmax(model)

lrp_name, lrp_kwargs, _ = u.Relevance_Method_LRP[0]
ig_name,  ig_kwargs,  _ = u.Relevance_Method_IG[0]

print('Creating LRP analyzer ...')
analyzer_lrp = innvestigate.create_analyzer(lrp_name, model_wo_sm, **lrp_kwargs)
print('Creating IG analyzer ...')
analyzer_ig  = innvestigate.create_analyzer(ig_name,  model_wo_sm, **ig_kwargs)

# ─── Load subjects ────────────────────────────────────────────────────────────
df_master = pd.read_csv(u.MASTER_CSV)
df_master = df_master[df_master['label'].isin(['CN', 'AD'])].reset_index(drop=True)
df_master = df_master[df_master['nifti_path'].apply(os.path.exists)].reset_index(drop=True)

TISSUE_REGIONS = ['CSF', 'GrayMatter', 'WhiteMatter']

records = []
errors  = []

print(f'Computing XAI agreement for {len(df_master)} subjects ...')

for i, row in df_master.iterrows():
    sid   = row['ID']
    label = row['label']
    print(f'  [{i+1}/{len(df_master)}] {sid} ({label})', end=' ... ')

    # Load FSL segmentation
    seg_path = u.find_fsl_seg_file(sid)
    if seg_path is None:
        errors.append(f'{sid}: No FSL_SEG file'); print('SKIP'); continue
    try:
        seg_vol = np.squeeze(nib.load(seg_path).get_fdata()).astype(int)
    except Exception as e:
        errors.append(f'{sid}: Seg error — {e}'); print('SKIP'); continue

    try:
        nifti_data = u.read_nifti(row['nifti_path'], minmax_scale=True)
        nifti_ds   = scipy.ndimage.zoom(nifti_data,
                         zoom=(44/176, 52/208, 44/176), order=1)

        # Compute LRP and IG maps
        act_lrp_ds = u.get_sample_activation(analyzer_lrp, nifti_ds, NODE)
        act_ig_ds  = u.get_sample_activation(analyzer_ig,  nifti_ds, NODE)

        # Upsample both back to full resolution
        act_lrp = scipy.ndimage.zoom(act_lrp_ds, zoom=(176/44, 208/52, 176/44), order=3)
        act_ig  = scipy.ndimage.zoom(act_ig_ds,  zoom=(176/44, 208/52, 176/44), order=3)

    except Exception as e:
        errors.append(f'{sid}: Activation error — {e}'); print('SKIP'); continue

    rec = {'SubjectID': sid, 'label': label}

    for tissue in TISSUE_REGIONS:
        mask = (seg_vol == u.FSL_LABELS[tissue])
        lrp_vals = act_lrp[mask].flatten()
        ig_vals  = act_ig[mask].flatten()

        if len(lrp_vals) < 10:
            rec[f'{tissue}_agree']   = 0.0
            rec[f'{tissue}_agree_p'] = 1.0
            continue

        # Spearman rank correlation: non-parametric, robust to scale differences
        # between LRP and IG (they have different value ranges)
        rho, pval = scipy.stats.spearmanr(lrp_vals, ig_vals)
        rho = 0.0 if np.isnan(rho) else float(rho)
        rec[f'{tissue}_agree']   = rho
        rec[f'{tissue}_agree_p'] = float(pval)

    rec['mean_agreement'] = float(np.mean([
        rec.get(f'{t}_agree', 0.0) for t in TISSUE_REGIONS]))

    records.append(rec)
    print(f'OK (mean rho={rec["mean_agreement"]:.3f})')

# ─── Save ─────────────────────────────────────────────────────────────────────
df_agree = pd.DataFrame(records)
out_path = os.path.join(SAVE_DIR, f'xai_agreement_node{NODE}.csv')
df_agree.to_csv(out_path, index=False)

print(f'\n✅ XAI agreement saved: {out_path}')
print(f'   Mean LRP-IG agreement across subjects:')
for t in TISSUE_REGIONS:
    col = f'{t}_agree'
    if col in df_agree.columns:
        print(f'     {t}: {df_agree[col].mean():.3f} ± {df_agree[col].std():.3f}')

if errors:
    print(f'⚠️  {len(errors)} errors:')
    for e in errors: print(f'   - {e}')
