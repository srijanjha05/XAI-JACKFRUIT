"""
STEP 3: Multi-Configuration Explanation Space Comparison + Clustering.

Evaluates 5 explanation space configurations via Agglomerative Ward Clustering,
mirroring and extending Table 2 of Singh et al. (2025):

  Config A: Basic LRP density only (replicating baseline)
  Config B: LRP density + spatial moments (mean, std, skew, kurt, entropy)
  Config C: Config B + HAI (Hemisphere Asymmetry Index)
  Config D: Config C + Concentration Ratio
  Config E: Config D + XAI agreement + nWBV (full context-enriched)

Novelty over Singh et al.:
  - Spatial moments as XAI features (not just density)
  - HAI: XAI-derived hemispheric asymmetry biomarker
  - Concentration Ratio: focal vs. diffuse attribution measure
  - XAI inter-method agreement as reliability feature
  - Applied to OASIS-1 (vs. 6-cohort multi-dataset in the paper)

Run with: python src/oasis_3_cluster_compare.py
"""

import os, warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.cluster import (v_measure_score, homogeneity_score,
                                     completeness_score, adjusted_rand_score,
                                     adjusted_mutual_info_score, silhouette_score,
                                     davies_bouldin_score)
from sklearn.metrics import fowlkes_mallows_score

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
K_CLUSTERS = 2
OUT_DIR  = '../4_clustering'
os.makedirs(OUT_DIR, exist_ok=True)

FEAT_PATH = f'../measures_combined/{EXP_NAME}/cv_{CV_FOLD}/{METHOD}/features_wscore_node{NODE}.csv'

print(f'Loading features: {FEAT_PATH}')
df = pd.read_csv(FEAT_PATH)

# ─── Feature group definitions ────────────────────────────────────────────────
# Config A: Basic LRP density (replicating Singh et al. basic explanation space)
FEATS_A = [c for c in df.columns if c.endswith('_w') and 'LRP_' in c and
           ('density' in c or '_pos' in c or '_neg' in c)]

# Config B: + Spatial moments (mean, std, skewness, kurtosis, entropy per tissue)
FEATS_B_extra = [c for c in df.columns if c.endswith('_w') and c.startswith('mom_')]

# Config C: + Hemisphere Asymmetry Index
FEATS_C_extra = [c for c in df.columns if c.endswith('_w') and c.startswith('hai_')]

# Config D: + Concentration Ratio
FEATS_D_extra = [c for c in df.columns if c.endswith('_w') and c.startswith('cr_')]

# Config E: + XAI agreement + nWBV volumetry (full context-enriched)
FEATS_E_extra = ([c for c in df.columns if '_agree' in c and '_p' not in c] +
                 [c for c in df.columns if 'nWBV' in c and c.endswith('_w')])

CONFIGS = {
    'A_BasicLRP':              FEATS_A,
    'B_LRP+Moments':           FEATS_A + FEATS_B_extra,
    'C_LRP+Moments+HAI':       FEATS_A + FEATS_B_extra + FEATS_C_extra,
    'D_LRP+Moments+HAI+CR':    FEATS_A + FEATS_B_extra + FEATS_C_extra + FEATS_D_extra,
    'E_FullEnriched':          FEATS_A + FEATS_B_extra + FEATS_C_extra + FEATS_D_extra + FEATS_E_extra,
}

print('\nFeature counts per configuration:')
for name, feats in CONFIGS.items():
    avail = [f for f in feats if f in df.columns]
    print(f'  {name}: {len(avail)} features')

# ─── Mutual information (feature relevance) ────────────────────────────────────
all_feature_candidates = list(set(sum(CONFIGS.values(), [])))
all_feature_candidates = [f for f in all_feature_candidates if f in df.columns]

df_clean = df[['SubjectID', 'label'] + all_feature_candidates].dropna()
y_raw    = df_clean['label']
le       = LabelEncoder()
y_int    = le.fit_transform(y_raw)

print(f'\nSubjects available: {len(df_clean)} (CN={sum(y_raw=="CN")}, AD={sum(y_raw=="AD")})')

X_all = df_clean[all_feature_candidates].values
mi    = mutual_info_classif(X_all, y_int, random_state=42)
mi_df = pd.DataFrame({'Feature': all_feature_candidates, 'MI': mi}).sort_values('MI', ascending=False)
mi_df.to_csv(os.path.join(OUT_DIR, 'feature_mutual_information.csv'), index=False)
print(f'\nTop-10 features by MI:')
print(mi_df.head(10).to_string(index=False))

# ─── Clustering for each configuration ────────────────────────────────────────
metrics_rows = []

for config_name, feat_list in CONFIGS.items():
    avail = [f for f in feat_list if f in df_clean.columns]
    if len(avail) == 0:
        print(f'\n[SKIP] {config_name} — no features available')
        continue

    print(f'\n{"─"*50}\n  Config: {config_name}  ({len(avail)} features)\n{"─"*50}')

    X = df_clean[avail].values
    # Standardize within each config (W-scores are already partially normalized,
    # but agreement and raw features benefit from scaling)
    X = StandardScaler().fit_transform(X)

    clust = AgglomerativeClustering(linkage='ward', n_clusters=K_CLUSTERS)
    labels_pred = clust.fit_predict(X)

    hom  = homogeneity_score(y_int, labels_pred)
    comp = completeness_score(y_int, labels_pred)
    vm   = v_measure_score(y_int, labels_pred)
    ari  = adjusted_rand_score(y_int, labels_pred)
    ami  = adjusted_mutual_info_score(y_int, labels_pred)
    fmi  = fowlkes_mallows_score(y_int, labels_pred)
    sil  = silhouette_score(X, labels_pred)
    dbs  = davies_bouldin_score(X, labels_pred)

    print(f'  Hom={hom:.3f}  Comp={comp:.3f}  V={vm:.3f}  ARI={ari:.3f}  '
          f'AMI={ami:.3f}  FMI={fmi:.3f}  Sil={sil:.3f}  DBI={dbs:.3f}')

    metrics_rows.append({
        'Config':              config_name,
        'N_Features':          len(avail),
        'Homogeneity':         round(hom, 4),
        'Completeness':        round(comp, 4),
        'V_measure':           round(vm, 4),
        'ARI':                 round(ari, 4),
        'AMI':                 round(ami, 4),
        'FMI':                 round(fmi, 4),
        'Silhouette':          round(sil, 4),
        'Davies_Bouldin':      round(dbs, 4),
    })

    # ── Clustermap ────────────────────────────────────────────────────────────
    df_plot = pd.DataFrame(X, columns=avail)
    row_colors = pd.Series(y_raw.values).map({'CN': 'steelblue', 'AD': 'crimson'}).values

    try:
        g = sns.clustermap(df_plot, method='ward', row_colors=row_colors,
                           cmap='RdBu_r', center=0,
                           figsize=(max(10, len(avail)), 10),
                           yticklabels=False,
                           col_cluster=(len(avail) > 1))
        g.ax_heatmap.set_title(f'Config {config_name}', fontsize=12)
        plt.savefig(os.path.join(OUT_DIR, f'clustermap_{config_name}.png'),
                    dpi=120, bbox_inches='tight')
        plt.close()
        print(f'  Clustermap saved.')
    except Exception as e:
        print(f'  [WARN] Clustermap failed: {e}')
        plt.close('all')

# ─── Save comparison table ────────────────────────────────────────────────────
df_cmp = pd.DataFrame(metrics_rows)
out_csv = os.path.join(OUT_DIR, 'config_comparison_metrics.csv')
df_cmp.to_csv(out_csv, index=False)
print(f'\n✅ Comparison table saved: {out_csv}')
print(df_cmp.to_string(index=False))

# Also save LRP_vs_IG_metrics.csv for backward compatibility
df_cmp.rename(columns={'Config': 'Method', 'N_Features': 'Selected_Features',
                        'ARI': 'Adjusted_Rand_Index'}).to_csv(
    os.path.join(OUT_DIR, 'LRP_vs_IG_metrics.csv'), index=False)

# ─── Bar chart: V-measure + ARI + Silhouette across configs ──────────────────
metrics_to_plot = ['Homogeneity', 'Completeness', 'V_measure', 'ARI', 'Silhouette']
x  = np.arange(len(metrics_to_plot))
w  = 0.15
colors = plt.cm.tab10(np.linspace(0, 0.6, len(df_cmp)))

fig, ax = plt.subplots(figsize=(12, 6))
for j, (_, row) in enumerate(df_cmp.iterrows()):
    vals = [row[m] for m in metrics_to_plot]
    ax.bar(x + j * w, vals, w, label=row['Config'], color=colors[j])

ax.set_xticks(x + w * (len(df_cmp) - 1) / 2)
ax.set_xticklabels(metrics_to_plot, fontsize=11)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Explanation Space Comparison — Clustering Quality\n'
             '(Extends Singh et al. 2025 Table 2)', fontsize=13)
ax.legend(loc='upper right', fontsize=8)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'config_comparison_chart.png'), dpi=150)
plt.close()
print('✅ Comparison bar chart saved.')

# ─── MI feature importance plot ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, max(6, len(mi_df)*0.25)))
top_n = min(20, len(mi_df))
mi_top = mi_df.head(top_n)
bars = ax.barh(range(top_n), mi_top['MI'].values, color='steelblue')
ax.set_yticks(range(top_n))
ax.set_yticklabels(mi_top['Feature'].values, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Mutual Information with AD/CN label')
ax.set_title(f'Top {top_n} XAI Features by Mutual Information')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'feature_importance_MI.png'), dpi=150)
plt.close()
print('✅ MI feature importance plot saved.')
