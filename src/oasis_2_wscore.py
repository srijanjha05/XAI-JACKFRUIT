"""
STEP 2: Compute W-scores for all XAI features (LRP density, spatial moments,
        HAI, concentration ratio) and merge into one combined feature CSV.

W-score formula (same as Singh et al. 2025, Eq. 1):
  W = (Observed - Expected_from_CN_model) / StdDev(CN_residuals)

Where Expected is predicted by a linear regression fitted on CN subjects only,
controlling for Age, Sex, eTIV (brain size confounders).

Extended over baseline:
  - LRP density features (existing)
  - Spatial moments: mean, std, skewness, kurtosis, entropy per tissue
  - Hemisphere Asymmetry Index (HAI) per tissue
  - Concentration Ratio (CR) per tissue

Run with: python src/oasis_2_wscore.py
"""

import os, pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def get_latest_exp():
    train_dir = '../train'
    exps = [d for d in os.listdir(train_dir)
            if 'OASIS1' in d and os.path.isdir(os.path.join(train_dir, d))]
    exps.sort(key=lambda x: os.path.getmtime(os.path.join(train_dir, x)))
    return exps[-1] if exps else 'REPLACE_ME'

# ─── Config ────────────────────────────────────────────────────────────────────
EXP_NAME   = get_latest_exp()
CV_FOLD    = 1
NODE       = 1
METHOD     = 'LRP-CMPalpha1beta0'

ACT_FILE   = f'../act_seg/{EXP_NAME}/cv_{CV_FOLD}/{METHOD}/SegmentationActivation_node{NODE}.xlsx'
AGREE_FILE = f'../act_seg/{EXP_NAME}/cv_{CV_FOLD}/xai_agreement_node{NODE}.csv'
MASTER     = '../data/oasis1_master.csv'
SAVE_DIR   = f'../measures_combined/{EXP_NAME}/cv_{CV_FOLD}/{METHOD}'
os.makedirs(SAVE_DIR, exist_ok=True)

REGIONS    = ['GrayMatter', 'WhiteMatter', 'CSF']
COVARIATES = ['Sex_int', 'Age', 'eTIV']


def compute_wscores(df, df_cn, feature_cols, covariates, prefix=''):
    """
    Compute W-scores for a set of feature columns.
    Fits linear regression on CN subjects; normalizes all subjects by CN residual std.
    Returns df_out with new columns named <prefix><col>_w for each col.
    """
    df_out = df.copy()
    model_dict = {}

    for col in feature_cols:
        if col not in df.columns:
            continue
        X_cn = df_cn[covariates].values
        y_cn = df_cn[col].values

        valid = ~np.isnan(y_cn) & ~np.any(np.isnan(X_cn), axis=1)
        if valid.sum() < 5:
            continue

        lr = LinearRegression().fit(X_cn[valid], y_cn[valid])
        residuals = y_cn[valid] - lr.predict(X_cn[valid])
        std_res = np.std(residuals)

        model_dict[col] = {'coef': lr.coef_, 'intercept': lr.intercept_, 'std_res': std_res}

        X_all = df[covariates].values
        y_all = df[col].values
        w_score = (y_all - lr.predict(X_all)) / (std_res if std_res > 0 else 1)
        df_out[f'{prefix}{col}_w'] = w_score

    return df_out, model_dict


# ─── Load activation data from Excel ─────────────────────────────────────────
print(f'Loading activation Excel: {ACT_FILE}')
xls = pd.ExcelFile(ACT_FILE)

# Standard density features
df_dens = pd.read_excel(xls, 'density_total', index_col=False)
df_dens.rename(columns={r: f'LRP_{r}_density' for r in REGIONS}, inplace=True)

df_pos = pd.read_excel(xls, 'act_sum_pos', index_col=False)
df_pos.rename(columns={r: f'LRP_{r}_pos' for r in REGIONS}, inplace=True)

df_neg = pd.read_excel(xls, 'act_sum_neg', index_col=False)
df_neg.rename(columns={r: f'LRP_{r}_neg' for r in REGIONS}, inplace=True)

# Spatial moments
df_moments = pd.read_excel(xls, 'spatial_moments', index_col=False)

# HAI
df_hai = pd.read_excel(xls, 'hai', index_col=False)
# Keep only HAI values (not raw L/R sums — they'll be W-scored)
hai_cols = ['SubjectID', 'label'] + [f'{r}_HAI' for r in REGIONS]
df_hai_filtered = df_hai[hai_cols] if all(c in df_hai.columns for c in hai_cols) else df_hai

# Concentration Ratio
df_cr = pd.read_excel(xls, 'concentration_ratio', index_col=False)

# ─── Merge all activation features ───────────────────────────────────────────
df_act = df_dens[['SubjectID', 'label'] + [f'LRP_{r}_density' for r in REGIONS if f'LRP_{r}_density' in df_dens.columns]]
df_act = pd.merge(df_act, df_pos[['SubjectID'] + [f'LRP_{r}_pos' for r in REGIONS if f'LRP_{r}_pos' in df_pos.columns]], on='SubjectID', how='left')
df_act = pd.merge(df_act, df_neg[['SubjectID'] + [f'LRP_{r}_neg' for r in REGIONS if f'LRP_{r}_neg' in df_neg.columns]], on='SubjectID', how='left')
df_act = pd.merge(df_act, df_moments.drop(columns=['label'], errors='ignore'), on='SubjectID', how='left')
df_act = pd.merge(df_act, df_hai_filtered.drop(columns=['label'], errors='ignore'), on='SubjectID', how='left')
df_act = pd.merge(df_act, df_cr.drop(columns=['label'], errors='ignore'), on='SubjectID', how='left')

print(f'Loaded activation data: {df_act.shape}')

# ─── Load XAI agreement scores (if available) ────────────────────────────────
if os.path.exists(AGREE_FILE):
    df_agree = pd.read_csv(AGREE_FILE)
    agree_cols = [c for c in df_agree.columns if '_agree' in c and '_p' not in c]
    df_act = pd.merge(df_act, df_agree[['SubjectID'] + agree_cols], on='SubjectID', how='left')
    print(f'Merged XAI agreement features: {agree_cols}')
else:
    print(f'[WARN] XAI agreement file not found: {AGREE_FILE}')
    print('       Run oasis_1b_xai_agreement.py first for full feature set.')

# ─── Load metadata ────────────────────────────────────────────────────────────
df_meta = pd.read_csv(MASTER)
df_meta = df_meta[df_meta['label'].isin(['CN', 'AD'])].copy()
df_meta['Sex_int'] = (df_meta['Sex'] == 'Female').astype(int)

def safe_numeric(s, vmin=None, vmax=None):
    s = pd.to_numeric(
        s.astype(str).str.extract(r'(-?\d+\.?\d*)', expand=False),
        errors='coerce')
    if vmin is not None: s = s.where(s >= vmin, other=np.nan)
    if vmax is not None: s = s.where(s <= vmax, other=np.nan)
    return s

df_meta['Age']  = safe_numeric(df_meta['Age'],  vmin=0, vmax=120)
df_meta['eTIV'] = safe_numeric(df_meta['eTIV'], vmin=800, vmax=3000)
df_meta['nWBV'] = safe_numeric(df_meta['nWBV'], vmin=0.5, vmax=1.0)
df_meta['MMSE'] = safe_numeric(df_meta['MMSE'], vmin=0, vmax=30)

df = pd.merge(df_act,
              df_meta[['ID', 'Sex_int', 'Age', 'eTIV', 'nWBV', 'MMSE']],
              left_on='SubjectID', right_on='ID', how='inner')
df.dropna(subset=COVARIATES, inplace=True)
print(f'After merge: {df.shape}  CN={sum(df["label"]=="CN")}  AD={sum(df["label"]=="AD")}')

df_cn = df[df['label'] == 'CN'].copy()

# ─── W-score: LRP density features ───────────────────────────────────────────
lrp_density_cols = [f'LRP_{r}_density' for r in REGIONS if f'LRP_{r}_density' in df.columns]
lrp_pos_cols     = [f'LRP_{r}_pos'     for r in REGIONS if f'LRP_{r}_pos'     in df.columns]
lrp_neg_cols     = [f'LRP_{r}_neg'     for r in REGIONS if f'LRP_{r}_neg'     in df.columns]
density_all = lrp_density_cols + lrp_pos_cols + lrp_neg_cols

df, model_density = compute_wscores(df, df_cn, density_all, COVARIATES)

# ─── W-score: Spatial Moments ─────────────────────────────────────────────────
moment_cols = [c for c in df.columns
               if any(c.startswith(r) and any(s in c for s in ['_mean','_std','_skew','_kurt','_entropy'])
                      for r in REGIONS)]
df, model_moments = compute_wscores(df, df_cn, moment_cols, COVARIATES, prefix='mom_')

# ─── W-score: HAI ─────────────────────────────────────────────────────────────
hai_cols_in_df = [f'{r}_HAI' for r in REGIONS if f'{r}_HAI' in df.columns]
df, model_hai = compute_wscores(df, df_cn, hai_cols_in_df, COVARIATES, prefix='hai_')

# ─── W-score: Concentration Ratio ────────────────────────────────────────────
cr_cols_in_df = [f'{r}_CR' for r in REGIONS if f'{r}_CR' in df.columns]
df, model_cr = compute_wscores(df, df_cn, cr_cols_in_df, COVARIATES, prefix='cr_')

# ─── W-score: Agreement (no covariates — it's already normalized) ─────────────
# Agreement is between [−1, 1]; just include as-is (no W-score needed)

# ─── W-score: nWBV (volumetric proxy) ────────────────────────────────────────
X_cn = df_cn[COVARIATES].values
y_cn = df_cn['nWBV'].values
valid = ~np.isnan(y_cn) & ~np.any(np.isnan(X_cn), axis=1)
lr_vol = LinearRegression().fit(X_cn[valid], y_cn[valid])
std_vol = np.std(y_cn[valid] - lr_vol.predict(X_cn[valid]))
df['nWBV_vol']   = df['nWBV']
df['nWBV_vol_w'] = (df['nWBV'].values - lr_vol.predict(df[COVARIATES].values)) / (std_vol if std_vol > 0 else 1)

# ─── Save combined model dict ─────────────────────────────────────────────────
all_models = {**model_density, **model_moments, **model_hai, **model_cr}
with open(os.path.join(SAVE_DIR, f'act_model_dict_node{NODE}.pkl'), 'wb') as f:
    pickle.dump(all_models, f)

# ─── Save combined feature CSV ────────────────────────────────────────────────
out_path = os.path.join(SAVE_DIR, f'features_wscore_node{NODE}.csv')
df.to_csv(out_path, index=False)

w_cols    = [c for c in df.columns if c.endswith('_w') or c.endswith('_vol') or '_agree' in c]
print(f'\n✅ Saved: {out_path}')
print(f'   W-score features ({len(w_cols)}): {w_cols}')
