"""
STEP 4: Rule-Based Textual Explanation Generator (NOVEL)

Generates per-patient clinical textual reports combining:
  - LRP relevance W-scores (CNN attribution evidence)
  - Spatial moment W-scores (focal vs. diffuse atrophy pattern)
  - HAI W-scores (hemisphere asymmetry — left vs. right dominance)
  - Concentration Ratio W-scores (spatial sparsity of attribution)
  - XAI agreement (LRP-IG consensus — reliability indicator)
  - nWBV volumetry W-score (structural atrophy measure)

Severity classification (extending Singh et al. rule-based approach):
  |W| 2-3 SD: mild pathology
  |W| 3-4 SD: moderate pathology
  |W| > 4 SD: strong pathology
  XAI agreement < 0.3: uncertain attribution (flag)

Novel contribution:
  - HAI-based lateralization in reports ("left-dominant", "right-dominant")
  - Concentration Ratio interpretation ("focal hotspot" vs "diffuse")
  - XAI agreement as explanation confidence qualifier

Outputs:
  reports/<exp>/textual_reports.csv  — per-patient feature summary
  reports/<exp>/reports.txt          — human-readable text reports
"""

import os
import numpy as np
import pandas as pd

def get_latest_exp():
    train_dir = '../train'
    exps = [d for d in os.listdir(train_dir)
            if 'OASIS1' in d and os.path.isdir(os.path.join(train_dir, d))]
    exps.sort(key=lambda x: os.path.getmtime(os.path.join(train_dir, x)))
    return exps[-1] if exps else 'REPLACE_ME'

EXP_NAME = get_latest_exp()
CV_FOLD  = 1
NODE     = 1
METHOD   = 'LRP-CMPalpha1beta0'

FEAT_PATH  = f'../measures_combined/{EXP_NAME}/cv_{CV_FOLD}/{METHOD}/features_wscore_node{NODE}.csv'
SCORE_PATH = f'../train/{EXP_NAME}/cv_{CV_FOLD}/test_set_scores_1.csv'
OUT_DIR    = f'../reports/{EXP_NAME}'
os.makedirs(OUT_DIR, exist_ok=True)

TISSUE_DISPLAY = {
    'GrayMatter':  'Gray Matter',
    'WhiteMatter': 'White Matter',
    'CSF':         'Cerebrospinal Fluid (CSF)',
}
SEVERITY_LABELS = {(2, 3): 'mild', (3, 4): 'moderate', (4, 999): 'strong'}


def w_severity(w_val):
    """Map W-score absolute value to severity string."""
    aw = abs(w_val)
    for (lo, hi), label in SEVERITY_LABELS.items():
        if lo <= aw < hi:
            return label
    return None   # below threshold (< 2 SD)


def hai_direction(hai_w):
    """Interpret HAI W-score as lateralization."""
    if abs(hai_w) < 1.5:
        return 'bilateral'
    return 'left-hemisphere dominant' if hai_w < 0 else 'right-hemisphere dominant'


def cr_pattern(cr_w):
    """Interpret Concentration Ratio W-score as focal/diffuse."""
    if cr_w > 1.5:
        return 'focal (concentrated hotspot)'
    elif cr_w < -1.5:
        return 'diffuse (distributed)'
    return 'intermediate'


def agreement_confidence(rho):
    """Map XAI agreement score to confidence label."""
    if rho >= 0.6:
        return 'high'
    elif rho >= 0.3:
        return 'moderate'
    else:
        return 'low (uncertain attribution)'


def generate_report(row, pred_row=None):
    """
    Generate a structured textual explanation for one patient.

    Follows the rule-based approach of Singh et al. (2025) but extends it with:
    - Hemisphere asymmetry (HAI) findings
    - Spatial concentration pattern (focal vs. diffuse)
    - XAI inter-method agreement as confidence qualifier
    """
    sid   = row.get('SubjectID', row.get('ID', 'Unknown'))
    label = row.get('label', 'Unknown')

    lines = []
    lines.append(f'Patient ID: {sid}')
    lines.append(f'True Diagnosis: {label}')

    if pred_row is not None:
        pred_label = pred_row.get('Pred_Label', 'N/A')
        ad_prob    = pred_row.get('AD_Prob', float('nan'))
        threshold  = pred_row.get('Threshold', 0.5)
        lines.append(f'CNN Prediction: {pred_label}  (AD probability: {ad_prob:.3f}, threshold: {threshold:.2f})')

    # Global summary
    lines.append('')
    lines.append('─── XAI Attribution Summary ───────────────────────────────')

    # XAI agreement (reliability)
    agreement_cols = [c for c in row.index if '_agree' in c and '_p' not in c]
    if agreement_cols:
        mean_agree = float(np.nanmean([row[c] for c in agreement_cols if not np.isnan(row.get(c, float('nan')))]))
        confidence = agreement_confidence(mean_agree)
        lines.append(f'XAI Reliability: {confidence} (LRP–IG Spearman ρ = {mean_agree:.3f})')
        if confidence.startswith('low'):
            lines.append('  ⚠ Low LRP-IG agreement: this subject shows atypical attribution pattern.')
            lines.append('    Clinical interpretation should be made with caution.')

    lines.append('')
    lines.append('─── Pathological Findings by Tissue Region ────────────────')

    found_any = False
    for tissue_key, tissue_name in TISSUE_DISPLAY.items():

        # Relevance density W-score
        dens_col = f'LRP_{tissue_key}_density_w'
        hai_col  = f'hai_{tissue_key}_HAI_w'
        cr_col   = f'cr_{tissue_key}_CR_w'
        mom_cols = {
            'mean':    f'mom_{tissue_key}_mean_w',
            'std':     f'mom_{tissue_key}_std_w',
            'entropy': f'mom_{tissue_key}_entropy_w',
        }

        dens_w = row.get(dens_col, float('nan'))
        if np.isnan(dens_w):
            continue

        sev = w_severity(dens_w)
        if sev is None:
            continue   # below 2 SD — not pathological

        found_any = True
        lines.append(f'\n  {tissue_name}:')
        lines.append(f'    Relevance W-score: {dens_w:+.2f}  [{sev.upper()} PATHOLOGY]')

        # Direction interpretation
        if dens_w > 0:
            lines.append(f'    → CNN showed ELEVATED relevance in {tissue_name} '
                         f'({"+".rstrip()} {dens_w:.2f} SD above CN norm).')
        else:
            lines.append(f'    → CNN showed REDUCED relevance in {tissue_name} '
                         f'({dens_w:.2f} SD below CN norm).')

        # HAI lateralization
        hai_w = row.get(hai_col, float('nan'))
        if not np.isnan(hai_w):
            lat = hai_direction(hai_w)
            if lat != 'bilateral':
                lines.append(f'    Lateralization: {lat}  (HAI W={hai_w:+.2f})')

        # Concentration pattern
        cr_w = row.get(cr_col, float('nan'))
        if not np.isnan(cr_w):
            pattern = cr_pattern(cr_w)
            lines.append(f'    Attribution pattern: {pattern}  (CR W={cr_w:+.2f})')

        # Spatial spread (entropy)
        ent_col = mom_cols['entropy']
        ent_w   = row.get(ent_col, float('nan'))
        if not np.isnan(ent_w):
            if ent_w > 1.5:
                lines.append(f'    Spatial spread: broadly distributed attribution (entropy W={ent_w:+.2f})')
            elif ent_w < -1.5:
                lines.append(f'    Spatial spread: highly concentrated attribution (entropy W={ent_w:+.2f})')

    if not found_any:
        lines.append('\n  No tissue regions exceeded the pathological threshold (|W| < 2 SD).')
        lines.append('  Attribution pattern consistent with Cognitively Normal profile.')

    # nWBV volumetric evidence
    nwbv_w = row.get('nWBV_vol_w', float('nan'))
    if not np.isnan(nwbv_w):
        lines.append('')
        lines.append('─── Volumetric Evidence ────────────────────────────────────')
        nwbv_sev = w_severity(nwbv_w)
        if nwbv_sev:
            lines.append(f'  Whole-brain volume (nWBV): W = {nwbv_w:+.2f}  [{nwbv_sev.upper()} ATROPHY]')
            if nwbv_w < 0:
                lines.append(f'  → Brain volume is {abs(nwbv_w):.1f} SD below age/sex/eTIV norm — consistent with AD atrophy.')
        else:
            lines.append(f'  Whole-brain volume (nWBV): W = {nwbv_w:+.2f}  [within normal range]')

    lines.append('')
    lines.append('=' * 65)
    return '\n'.join(lines)


# ─── Load data ────────────────────────────────────────────────────────────────
print(f'Loading features: {FEAT_PATH}')
df = pd.read_csv(FEAT_PATH)

# Load predictions if available
df_preds = None
if os.path.exists(SCORE_PATH):
    df_preds = pd.read_csv(SCORE_PATH)
    df_preds['SubjectID'] = df_preds['Filename'].apply(
        lambda p: os.path.splitext(os.path.splitext(os.path.basename(p))[0])[0])
    print(f'Loaded predictions: {len(df_preds)} subjects')

# ─── Generate reports ─────────────────────────────────────────────────────────
all_reports = []
records     = []

for _, row in df.iterrows():
    sid = row.get('SubjectID', '')

    pred_row = None
    if df_preds is not None:
        matches = df_preds[df_preds['SubjectID'] == sid]
        if len(matches) > 0:
            pred_row = matches.iloc[0].to_dict()

    report_text = generate_report(row, pred_row)
    all_reports.append(report_text)

    # Summary record for CSV
    rec = {'SubjectID': sid, 'label': row.get('label', 'Unknown')}
    if pred_row:
        rec['pred_label'] = pred_row.get('Pred_Label', '')
        rec['ad_prob']    = pred_row.get('AD_Prob', np.nan)

    for tissue_key in TISSUE_DISPLAY:
        dens_col = f'LRP_{tissue_key}_density_w'
        hai_col  = f'hai_{tissue_key}_HAI_w'
        cr_col   = f'cr_{tissue_key}_CR_w'
        for col in [dens_col, hai_col, cr_col]:
            rec[col] = row.get(col, np.nan)
        sev = w_severity(row.get(dens_col, float('nan'))) if not np.isnan(row.get(dens_col, float('nan'))) else None
        rec[f'{tissue_key}_severity'] = sev or 'normal'

    rec['nWBV_w']    = row.get('nWBV_vol_w', np.nan)
    agree_cols = [c for c in row.index if '_agree' in c and '_p' not in c]
    rec['mean_xai_agreement'] = float(np.nanmean([row[c] for c in agree_cols if not np.isnan(row.get(c, float('nan')))])) if agree_cols else np.nan
    records.append(rec)

# ─── Save outputs ─────────────────────────────────────────────────────────────
df_summary = pd.DataFrame(records)
df_summary.to_csv(os.path.join(OUT_DIR, 'textual_reports_summary.csv'), index=False)

with open(os.path.join(OUT_DIR, 'reports.txt'), 'w') as f:
    f.write('\n\n'.join(all_reports))

print(f'\n✅ Reports saved to: {OUT_DIR}')
print(f'   {len(all_reports)} patient reports generated')

# Print 2 example reports (one CN, one AD)
print('\n' + '='*65)
print('EXAMPLE REPORTS:')
print('='*65)
ad_example = df_summary[df_summary['label'] == 'AD'].iloc[0]['SubjectID'] if sum(df_summary['label'] == 'AD') > 0 else None
cn_example = df_summary[df_summary['label'] == 'CN'].iloc[0]['SubjectID'] if sum(df_summary['label'] == 'CN') > 0 else None

for sid_ex in [ad_example, cn_example]:
    if sid_ex is None:
        continue
    row_ex = df[df['SubjectID'] == sid_ex].iloc[0]
    pred_ex = None
    if df_preds is not None:
        m = df_preds[df_preds['SubjectID'] == sid_ex]
        if len(m) > 0:
            pred_ex = m.iloc[0].to_dict()
    print(generate_report(row_ex, pred_ex))
    print()
