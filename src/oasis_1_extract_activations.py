"""
STEP 0b: Extract LRP/IG activations per FSL tissue region for all OASIS-1 subjects.
Run AFTER oasis_0_train_cnn.py (needs a trained .h5 model).
Run with: ~/miniconda3/envs/xai4dem/bin/python src/oasis_1_extract_activations.py

Outputs:
  act_seg/<exp_name>/cv_<fold>/<method>/SegmentationActivation_node<N>.xlsx
  (one Excel file with sheets: act_sum_pos, act_sum_neg, act_total, density_pos, density_neg, density_total, no_voxels)
"""

import os, logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').disabled  = True
logging.getLogger('innvestigate').disabled = True

import numpy as np
import pandas as pd
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import innvestigate
from keras.models import load_model
import oasis_util as u

def get_latest_exp():
    train_dir = '../train'
    exps = [d for d in os.listdir(train_dir) if 'OASIS1' in d and os.path.isdir(os.path.join(train_dir, d))]
    exps.sort(key=lambda x: os.path.getmtime(os.path.join(train_dir, x)))
    return exps[-1] if exps else 'REPLACE_ME'

# ─── Config — match what was used in training ─────────────────────────────────
EXP_NAME   = get_latest_exp()
CV_FOLD    = 1                               # which fold's model to use
NODE       = 1                               # 0=CN, 1=AD  (explain AD predictions)
METHOD_IDX = 0                              # 0 = first entry (now IG in util)

MODEL_PATH = f'../train/{EXP_NAME}/cv_{CV_FOLD}/ADvsCN_cv{CV_FOLD:02d}.h5'
SAVE_DIR   = f'../act_seg/{EXP_NAME}/cv_{CV_FOLD}/{u.Relevance_Method[METHOD_IDX][2]}'
os.makedirs(SAVE_DIR, exist_ok=True)

# ─── Load model + create analyzer ────────────────────────────────────────────
print(f'Loading model: {MODEL_PATH}')
model = load_model(MODEL_PATH)
model_wo_sm = innvestigate.model_wo_softmax(model)
analyzer = innvestigate.create_analyzer(
    u.Relevance_Method[METHOD_IDX][0],
    model_wo_sm,
    **u.Relevance_Method[METHOD_IDX][1]
)

# ─── Load subject list ────────────────────────────────────────────────────────
df_master = pd.read_csv(u.MASTER_CSV)
df_master = df_master[df_master['label'].isin(['CN', 'AD'])].reset_index(drop=True)
df_master = df_master[df_master['nifti_path'].apply(os.path.exists)].reset_index(drop=True)

regions = list(u.FSL_LABELS.keys())   # ['Background', 'CSF', 'GrayMatter', 'WhiteMatter']
cols = ['SubjectID', 'label'] + regions

df_act_pos   = pd.DataFrame(columns=cols)
df_act_neg   = pd.DataFrame(columns=cols)
df_act_total = pd.DataFrame(columns=cols)
df_dens_pos  = pd.DataFrame(columns=cols)
df_dens_neg  = pd.DataFrame(columns=cols)
df_dens_tot  = pd.DataFrame(columns=cols)
df_voxels    = pd.DataFrame(columns=cols)

errors = []

print(f'Processing {len(df_master)} subjects | Node: {NODE} | Method: {u.Relevance_Method[METHOD_IDX][2]}')

for i, row in df_master.iterrows():
    sid   = row['ID']
    label = row['label']
    print(f'  [{i+1}/{len(df_master)}] {sid} ({label})')

    # 1. Load FSL segmentation
    seg_path = u.find_fsl_seg_file(sid)
    if seg_path is None:
        errors.append(f'{sid}: No FSL_SEG file found')
        continue
    try:
        import nibabel as nib
        seg_vol = np.squeeze(nib.load(seg_path).get_fdata()).astype(int)
    except Exception as e:
        errors.append(f'{sid}: Seg load error — {e}')
        continue

    # 2. Compute LRP/IG activation map
    try:
        import scipy.ndimage
        nifti_data = u.read_nifti(row['nifti_path'], minmax_scale=True)
        # downsample to match model input (88, 104, 88)
        nifti_ds = scipy.ndimage.zoom(nifti_data, zoom=[1/2]*3, order=1)
        
        activation_ds = u.get_sample_activation(analyzer, nifti_ds, NODE)
        # upsample activation back to (176, 208, 176)
        activation = scipy.ndimage.zoom(activation_ds, zoom=[2]*3, order=1)
    except Exception as e:
        errors.append(f'{sid}: Activation error — {e}')
        continue

    # 3. Extract per-region statistics
    row_pos, row_neg, row_tot = [sid, label], [sid, label], [sid, label]
    row_dp,  row_dn, row_dt  = [sid, label], [sid, label], [sid, label]
    row_vox = [sid, label]

    for region, label_code in u.FSL_LABELS.items():
        mask    = (seg_vol == label_code)
        n_vox   = int(np.sum(mask))
        seg_act = activation * mask

        ap = u.sign_sum(seg_act, 'pos')
        an = u.sign_sum(seg_act, 'neg')
        at = ap + an

        row_pos.append(ap);  row_neg.append(an);  row_tot.append(at)
        row_vox.append(n_vox)
        row_dp.append(ap / n_vox if n_vox > 0 else 0)
        row_dn.append(an / n_vox if n_vox > 0 else 0)
        row_dt.append(at / n_vox if n_vox > 0 else 0)

    df_act_pos.loc[len(df_act_pos)]   = row_pos
    df_act_neg.loc[len(df_act_neg)]   = row_neg
    df_act_total.loc[len(df_act_total)]= row_tot
    df_dens_pos.loc[len(df_dens_pos)] = row_dp
    df_dens_neg.loc[len(df_dens_neg)] = row_dn
    df_dens_tot.loc[len(df_dens_tot)] = row_dt
    df_voxels.loc[len(df_voxels)]     = row_vox

# ─── Save to Excel ────────────────────────────────────────────────────────────
out_file = os.path.join(SAVE_DIR, f'SegmentationActivation_node{NODE}.xlsx')
with pd.ExcelWriter(out_file) as writer:
    df_act_pos.to_excel(writer,   sheet_name='act_sum_pos', index=False)
    df_act_neg.to_excel(writer,   sheet_name='act_sum_neg', index=False)
    df_act_total.to_excel(writer, sheet_name='act_total',   index=False)
    df_dens_pos.to_excel(writer,  sheet_name='density_pos', index=False)
    df_dens_neg.to_excel(writer,  sheet_name='density_neg', index=False)
    df_dens_tot.to_excel(writer,  sheet_name='density_total',index=False)
    df_voxels.to_excel(writer,    sheet_name='no_voxels',   index=False)

print(f'\n✅ Saved: {out_file}')
if errors:
    print(f'⚠️  {len(errors)} errors:')
    for e in errors: print(f'   - {e}')
    with open(os.path.join(SAVE_DIR, f'errors_node{NODE}.txt'), 'w') as f:
        f.writelines(line + '\n' for line in errors)
