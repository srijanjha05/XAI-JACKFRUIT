"""
STEP 3+4: Feature selection (Mutual Information) + Agglomerative Ward Clustering.
Compares LRP vs IG clustering quality.
Run with: ~/miniconda3/envs/xai4dem/bin/python src/oasis_3_cluster_compare.py

Inputs:
  measures_combined/<exp>/cv_<fold>/LRP-CMPalpha1beta0/features_wscore_node<N>.csv
  measures_combined/<exp>/cv_<fold>/IG/features_wscore_node<N>.csv

Outputs:
  4_clustering/  — clustermaps and metrics comparison table (LRP vs IG)
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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.cluster import (v_measure_score, homogeneity_score,
                                     completeness_score, adjusted_rand_score,
                                     silhouette_score, davies_bouldin_score)

def get_latest_exp():
    train_dir = '../train'
    exps = [d for d in os.listdir(train_dir) if 'OASIS1' in d and os.path.isdir(os.path.join(train_dir, d))]
    exps.sort(key=lambda x: os.path.getmtime(os.path.join(train_dir, x)))
    return exps[-1] if exps else 'REPLACE_ME'

# ─── Config ────────────────────────────────────────────────────────────────────
EXP_NAME = get_latest_exp()
CV_FOLD  = 1
NODE     = 1
METHODS  = ['LRP-CMPalpha1beta0']   # only running LRP
MI_THRESHOLD = -1.0                        # use ALL extracted LRP features!
K_CLUSTERS   = 2                           # CN-stable vs AD-converter
OUT_DIR = '../4_clustering'
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Feature columns ───────────────────────────────────────────────────────────
REL_FEATURES = [
    'GrayMatter_density_rel', 'GrayMatter_pos_rel', 'GrayMatter_neg_rel',
    'WhiteMatter_density_rel', 'WhiteMatter_pos_rel', 'WhiteMatter_neg_rel',
    'CSF_density_rel', 'CSF_pos_rel', 'CSF_neg_rel'
]
VOL_FEATURES = ['nWBV_vol_w']
ALL_FEATURES = REL_FEATURES + VOL_FEATURES

# ─── Load and process one method ──────────────────────────────────────────────
def load_features(method):
    path = f'../measures_combined/{EXP_NAME}/cv_{CV_FOLD}/{method}/features_wscore_node{NODE}.csv'
    df   = pd.read_csv(path)
    # Keep only rows with all features present
    available = [f for f in ALL_FEATURES if f in df.columns]
    df = df[['SubjectID', 'label'] + available].dropna()
    X  = df[available]
    y  = df['label']
    return X, y, available

# ─── Run for each method ───────────────────────────────────────────────────────
metrics_rows = []

for method in METHODS:
    print(f'\n{"="*50}\n  Method: {method}\n{"="*50}')
    try:
        X, y, features = load_features(method)
    except FileNotFoundError:
        print(f'  [SKIP] File not found — run oasis_2_wscore.py for {method} first')
        continue

    y_int = LabelEncoder().fit_transform(y)   # CN=0, AD=1 (or vice versa) for metrics

    # ── 1. Mutual Information feature selection ────────────────────────────────
    mi = mutual_info_classif(X, y, random_state=42)
    mi_dict = dict(zip(features, mi))
    print(f'  MI scores: {mi_dict}')
    selected = [f for f, score in mi_dict.items() if score >= MI_THRESHOLD]
    if not selected:
        selected = features   # use all if none pass threshold
    print(f'  Selected features (MI>{MI_THRESHOLD}): {selected}')

    # ── 2. Agglomerative Ward Clustering ──────────────────────────────────────
    X_sel = X[selected].values
    model = AgglomerativeClustering(linkage='ward', n_clusters=K_CLUSTERS)
    labels_pred = model.fit_predict(X_sel)

    hom  = homogeneity_score(y_int, labels_pred)
    comp = completeness_score(y_int, labels_pred)
    vm   = v_measure_score(y_int, labels_pred)
    ari  = adjusted_rand_score(y_int, labels_pred)
    sil  = silhouette_score(X_sel, labels_pred)
    dbs  = davies_bouldin_score(X_sel, labels_pred)

    metrics_rows.append({
        'Method':              method,
        'Selected_Features':   len(selected),
        'Homogeneity':         round(hom, 4),
        'Completeness':        round(comp, 4),
        'V_measure':           round(vm, 4),
        'Adjusted_Rand_Index': round(ari, 4),
        'Silhouette':          round(sil, 4),
        'Davies_Bouldin':      round(dbs, 4),
    })
    print(f'  Hom={hom:.3f}  Comp={comp:.3f}  V={vm:.3f}  ARI={ari:.3f}  Sil={sil:.3f}')

    # ── 3. Clustermap ──────────────────────────────────────────────────────────
    df_plot = X[selected].copy()
    df_plot.index = y.values   # label as index for row colors
    row_colors = pd.Series(y.values).map({'CN': 'steelblue', 'AD': 'crimson'}).values

    cluster_cols = len(selected) > 1
    g = sns.clustermap(df_plot, method='ward', row_colors=row_colors,
                       cmap='RdBu_r', center=0, figsize=(max(8, len(selected)*2), 10),
                       yticklabels=False, col_cluster=cluster_cols)
    g.ax_heatmap.set_title(f'Clustermap — {method}', fontsize=14)
    plt.savefig(os.path.join(OUT_DIR, f'clustermap_{method}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Clustermap saved.')

# ─── Save comparison table ────────────────────────────────────────────────────
if metrics_rows:
    df_cmp = pd.DataFrame(metrics_rows)
    out_csv = os.path.join(OUT_DIR, 'LRP_vs_IG_metrics.csv')
    df_cmp.to_csv(out_csv, index=False)
    print(f'\n✅ Comparison table saved: {out_csv}')
    print(df_cmp.to_string(index=False))

    # ── Bar chart comparison ────────────────────────────────────────────────
    metrics_to_plot = ['Homogeneity', 'Completeness', 'V_measure', 'Adjusted_Rand_Index']
    x   = np.arange(len(metrics_to_plot))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#d62728', '#1f77b4']

    for j, row in enumerate(df_cmp.itertuples()):
        vals = [getattr(row, m) for m in metrics_to_plot]
        ax.bar(x + j*w, vals, w, label=row.Method, color=colors[j % len(colors)])

    ax.set_xticks(x + w/2)
    ax.set_xticklabels(metrics_to_plot)
    ax.set_ylabel('Score')
    ax.set_title('LRP vs Integrated Gradients — Clustering Quality')
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'LRP_vs_IG_comparison_chart.png'), dpi=150)
    plt.close()
    print('✅ Comparison bar chart saved.')
else:
    print('\n⚠️  No results — make sure both LRP and IG excel files exist.')
