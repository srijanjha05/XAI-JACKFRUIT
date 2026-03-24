#!/bin/bash
# Master Runner Script for OASIS-1 XAI4Dementia Pipeline
# Run this from the xai4dementia-framework/src directory

set -e

# Make sure we're using the right conda environment
# (Assuming the user will run this within the activated environment, 
#  but we can explicitly use the full path just in case)
PYTHON_BIN="$HOME/miniconda3/envs/xai4dem/bin/python"

echo "========================================================="
echo "  XAI4Dementia OASIS-1 Pipeline - Full Run"
echo "========================================================="

cd "$(dirname "$0")"

# STEP 0: Train 3D CNN (saves LRP and IG heatmaps automatically)
echo "---------------------------------------------------------"
echo "STEP 0: Training 3D DenseNet CNN (CN vs AD)"
echo "---------------------------------------------------------"
# $PYTHON_BIN oasis_0_train_cnn.py

# Extract the experiment name that was just generated
# We look in the ../train directory for the most recent folder
EXP_NAME=$(ls -td ../train/*_OASIS1_CNvAD | head -1 | xargs basename)
echo "Experiment Name detected: $EXP_NAME"

# STEP 1: Extract Activations (LRP)
echo "---------------------------------------------------------"
echo "STEP 1: Extracting LRP Activations"
echo "---------------------------------------------------------"
$PYTHON_BIN oasis_1_extract_activations.py

# STEP 2: Compute W-Scores (LRP)
echo "---------------------------------------------------------"
echo "STEP 2: Computing W-Scores for LRP"
echo "---------------------------------------------------------"
$PYTHON_BIN oasis_2_wscore.py

# STEP 3 & 4: Feature Selection & Clustering (Comparison)
# STEP 3: Feature Selection & Clustering
echo "---------------------------------------------------------"
echo "STEP 3: Clustering Evaluation"
echo "---------------------------------------------------------"
$PYTHON_BIN oasis_3_cluster_compare.py

# Cleanup backups
rm -f *.bak

echo "========================================================="
echo "✅ PIPELINE COMPLETE!"
echo "Main Results: ../4_clustering/LRP_vs_IG_metrics.csv"
echo "========================================================="
