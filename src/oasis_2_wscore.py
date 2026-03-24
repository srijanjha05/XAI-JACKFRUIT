"""
STEP 2: Compute w-scores for LRP activations + volumetric features,
        then merge into one combined feature CSV.
Run with: ~/miniconda3/envs/xai4dem/bin/python src/oasis_2_wscore.py

Inputs:
  act_seg/<exp>/cv_<fold>/<method>/SegmentationActivation_node<N>.xlsx
  data/oasis1_master.csv

Outputs:
  measures_combined/<exp>/cv_<fold>/<method>/features_wscore_node<N>.csv
"""

import os, pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def get_latest_exp():
    train_dir = '../train'
    exps = [d for d in os.listdir(train_dir) if 'OASIS1' in d and os.path.isdir(os.path.join(train_dir, d))]
    exps.sort(key=lambda x: os.path.getmtime(os.path.join(train_dir, x)))
    return exps[-1] if exps else 'REPLACE_ME'

# ─── Config — update with your exp name after training ────────────────────────
EXP_NAME  = get_latest_exp()
CV_FOLD   = 1
NODE      = 1        # 1=AD node
METHOD    = 'LRP-CMPalpha1beta0'   # or 'IG'

ACT_FILE  = f'../act_seg/{EXP_NAME}/cv_{CV_FOLD}/{METHOD}/SegmentationActivation_node{NODE}.xlsx'
MASTER    = '../data/oasis1_master.csv'
SAVE_DIR  = f'../measures_combined/{EXP_NAME}/cv_{CV_FOLD}/{METHOD}'
os.makedirs(SAVE_DIR, exist_ok=True)

REGIONS   = ['GrayMatter', 'WhiteMatter', 'CSF']   # skip Background
COVARIATES= ['Sex_int', 'Age', 'eTIV']              # numeric covariates

# ─── Load activation data ─────────────────────────────────────────────────────
xls         = pd.ExcelFile(ACT_FILE)

# Load density
df_dens = pd.read_excel(xls, 'density_total', index_col=False)
df_dens.rename(columns={r: f'LRP_{r}_density' for r in ['GrayMatter', 'WhiteMatter', 'CSF']}, inplace=True)

# Load pos
df_pos = pd.read_excel(xls, 'act_sum_pos', index_col=False)
df_pos.rename(columns={r: f'LRP_{r}_pos' for r in ['GrayMatter', 'WhiteMatter', 'CSF']}, inplace=True)

# Load neg
df_neg = pd.read_excel(xls, 'act_sum_neg', index_col=False)
df_neg.rename(columns={r: f'LRP_{r}_neg' for r in ['GrayMatter', 'WhiteMatter', 'CSF']}, inplace=True)

# Merge them
df_act = pd.merge(df_dens, df_pos[['SubjectID', 'LRP_GrayMatter_pos', 'LRP_WhiteMatter_pos', 'LRP_CSF_pos']], on='SubjectID')
df_act = pd.merge(df_act, df_neg[['SubjectID', 'LRP_GrayMatter_neg', 'LRP_WhiteMatter_neg', 'LRP_CSF_neg']], on='SubjectID')

print(f'Loaded activation data: {df_act.shape}')

# ─── Load + merge metadata ────────────────────────────────────────────────────
df_meta = pd.read_csv(MASTER)
df_meta = df_meta[df_meta['label'].isin(['CN', 'AD'])].copy()
df_meta['Sex_int'] = (df_meta['Sex'] == 'Female').astype(int)
df_meta['Age']     = pd.to_numeric(df_meta['Age'],  errors='coerce')
df_meta['eTIV']    = pd.to_numeric(df_meta['eTIV'], errors='coerce')
df_meta['nWBV']    = pd.to_numeric(df_meta['nWBV'], errors='coerce')
df_meta['MMSE']    = pd.to_numeric(df_meta['MMSE'], errors='coerce')

# exclude 'label' from df_meta to prevent label_x/label_y suffixing
df = pd.merge(df_act, df_meta[['ID', 'Sex_int', 'Age', 'eTIV', 'nWBV', 'MMSE']],
              left_on='SubjectID', right_on='ID', how='inner')
df.dropna(subset=COVARIATES, inplace=True)
print(f'After merge: {df.shape}  CN={sum(df["label"]=="CN")}  AD={sum(df["label"]=="AD")}')

# ─── Compute w-scores for activation features ─────────────────────────────────
# W-score = (measured - predicted_by_covariates_from_CN_model) / std_residual_of_CN
df_cn = df[df['label'] == 'CN'].copy()
act_model_dict = {}

df_w = df.copy()

for region in REGIONS:
    for stat in ['density', 'pos', 'neg']:
        if stat == 'density':
            col_name = region # backward compatibility: 'density' was aliased to Region name 
        else:
            col_name = f'LRP_{region}_{stat}'

        if col_name not in df.columns:
            print(f'  [SKIP] {col_name} not in data')
            continue

        X_cn = df_cn[COVARIATES].values
        y_cn = df_cn[col_name].values

        lr = LinearRegression().fit(X_cn, y_cn)
        residuals = y_cn - lr.predict(X_cn)
        std_res = np.std(residuals)

        act_model_dict[col_name] = {
            'coef': lr.coef_, 'intercept': lr.intercept_, 'std_res': std_res
        }

        X_all = df[COVARIATES].values
        y_all = df[col_name].values
        w_score = (y_all - lr.predict(X_all)) / (std_res if std_res > 0 else 1)
        df_w[f'{region}_{stat}_rel'] = w_score

# Save model dict
with open(os.path.join(SAVE_DIR, f'act_model_dict_node{NODE}.pkl'), 'wb') as f:
    pickle.dump(act_model_dict, f)

# ─── Use nWBV as volumetry proxy (global brain volume w-score) ────────────────
# (Replaces per-ROI volumetry since we don't have FreeSurfer ROI stats)
# nWBV = normalized whole-brain volume — already a ratio, use as-is
df_w['nWBV_vol'] = df['nWBV']

# Also compute w-score for nWBV
X_cn = df_cn[COVARIATES].values
y_cn = df_cn['nWBV'].values
lr_vol = LinearRegression().fit(X_cn, y_cn)
std_vol = np.std(y_cn - lr_vol.predict(X_cn))
df_w['nWBV_vol_w'] = (df['nWBV'].values - lr_vol.predict(df[COVARIATES].values)) / (std_vol if std_vol > 0 else 1)

# ─── Save combined feature CSV ────────────────────────────────────────────────
out_path = os.path.join(SAVE_DIR, f'features_wscore_node{NODE}.csv')
df_w.to_csv(out_path, index=False)
print(f'\n✅ Saved: {out_path}')
print(f'   Columns: {[c for c in df_w.columns if "_rel" in c or "_vol" in c]}')
