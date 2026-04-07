#!/bin/bash
# Master Runner — OASIS-1 XAI4Alzheimer Pipeline
# Novel extended pipeline with spatial XAI features + inter-method agreement
# Run from: xai4dementia-framework/src/

set -e
PYTHON_BIN="$HOME/miniconda3/envs/xai4dem/bin/python"

echo "========================================================="
echo "  XAI4Alzheimer OASIS-1 Pipeline — Extended Novel Run"
echo "========================================================="

cd "$(dirname "$0")"

echo "---------------------------------------------------------"
echo "STEP 0: Train 3D Multi-Modal DenseNet (CN vs AD)"
echo "        [focal loss + balanced batches + threshold opt]"
echo "---------------------------------------------------------"
$PYTHON_BIN oasis_0_train_cnn.py

echo "---------------------------------------------------------"
echo "STEP 1: Extract LRP + IG attribution maps"
echo "        [+ spatial moments, HAI, concentration ratio]"
echo "---------------------------------------------------------"
$PYTHON_BIN oasis_1_extract_activations.py

echo "---------------------------------------------------------"
echo "STEP 1b: Multi-XAI Inter-Method Agreement Scoring (NOVEL)"
echo "         [LRP vs IG Spearman correlation per tissue]"
echo "---------------------------------------------------------"
$PYTHON_BIN oasis_1b_xai_agreement.py

echo "---------------------------------------------------------"
echo "STEP 2: W-Score normalization (all feature groups)"
echo "---------------------------------------------------------"
$PYTHON_BIN oasis_2_wscore.py

echo "---------------------------------------------------------"
echo "STEP 3: Multi-Config Explanation Space Comparison"
echo "        [mirrors + extends Singh et al. 2025 Table 2]"
echo "---------------------------------------------------------"
$PYTHON_BIN oasis_3_cluster_compare.py

echo "---------------------------------------------------------"
echo "STEP 4: Rule-Based Textual Reports (NOVEL)"
echo "        [LRP + HAI + CR + agreement → per-patient text]"
echo "---------------------------------------------------------"
$PYTHON_BIN oasis_4_textual_report.py

rm -f *.bak

echo "========================================================="
echo "✅  PIPELINE COMPLETE!"
echo ""
echo "  Key outputs:"
echo "  • Accuracy + AUC:  ../train/<exp>/test_set_metrics.csv"
echo "  • XAI features:    ../act_seg/<exp>/cv_1/LRP-*/SegmentationActivation_node1.xlsx"
echo "  • XAI agreement:   ../act_seg/<exp>/cv_1/xai_agreement_node1.csv"
echo "  • W-scored feats:  ../measures_combined/<exp>/cv_1/LRP-*/features_wscore_node1.csv"
echo "  • Clustering:      ../4_clustering/config_comparison_metrics.csv"
echo "  • Text reports:    ../reports/<exp>/reports.txt"
echo "========================================================="
