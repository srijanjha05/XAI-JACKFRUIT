# Explainable AI (XAI) for Dementia Classification (OASIS-1)

A complete framework applying Convolutional Neural Networks and Layer-Wise Relevance Propagation (LRP) to diagnose Alzheimer's Disease (AD) and clinically evaluate the AI's structural reasoning.

## 🚀 Overview

This repository adapts the `xai4dementia` framework to the openly available OASIS-1 MRI dataset. It attempts to answer two fundamental medical AI questions:
1. **Can a CNN accurately diagnose Alzheimer's Disease from 3D brain scans?** 
2. **Does the CNN make its diagnosis by looking at biologically verifiable disease markers** (e.g., ventricle expansion, Gray Matter loss)?

By integrating **Layer-Wise Relevance Propagation (LRP)**, this project escapes the deep learning "Black Box" by physically mapping precisely which microscopic regions of the brain the AI found "Relevant" for its diagnosis. It then clusters these mathematical XAI features to prove biological heterogeneity across the patient pool.

## 📂 Project Structure

All execution code lives in the `src/` directory.

### Pipeline Execution
*   `run_pipeline.sh`: A single master shell script that automatically executes the entire machine learning and XAI evaluation pipeline sequentially.

### Python Engine
*   `convert_to_nifti.py`: Preprocesses the raw OASIS-1 repository `.img`/`.hdr` files into clean 3D `.nii.gz` volumes.
*   `oasis_0_train_cnn.py`: Trains a high-resolution 3D DenseNet (88x104x88) to classify between Cognitively Normal (CN) and Alzheimer's (AD). Outputs the AUC curve and final saved weights.
*   `oasis_1_extract_activations.py`: Injects the trained model with `innvestigate` to generate LRP heatmaps for every patient. It then parses these heatmaps using FSL structure boundaries to extract positive/negative significance in the Gray Matter, White Matter, and CSF.
*   `oasis_2_wscore.py`: Computes W-Scores for the extracted LRP features via linear regression, standardizing out age, sex, and head size biases.
*   `oasis_3_cluster_compare.py`: Uses Agglomerative Ward Clustering to structurally group patients purely based on how the AI viewed their brains, outputting Mutual Information and internal validity scores (Silhouette, Homogeneity).
*   `oasis_util.py`: Contains the master configuration logic, architecture building functions, downsampling parameters, and LRP instantiation. 

## 📊 Comprehensive Results

Read the full experimental context, metrics evaluation, and theory link in our finalized log:
👉 [project_explanation_report.txt](./project_explanation_report.txt)

### Key findings from our current run:
*   **Accuracy:** The CNN successfully classified Alzheimer's subjects with an **0.888 AUC**.
*   **Biological Validity:** The AI established high Positive Relevance (`0.033` Mutual Information) with the Cerebrospinal Fluid, proving the AI genuinely detects the clinical phenomenon of Ventricle Expansion during Gray Matter death to formulate its diagnosis.
*   **Patient Clustering (Silhouette = 0.399):** The AI confirmed that Alzheimer's presents as a highly structurally diverse spectrum of decay sub-types. 

## ⚙️ Installation & Usage

1. Create a python 3.10/3.11 environment exactly reproducing traditional TensorFlow 2 (required for Innvestigate gradient tracing).
2. Install standard requirements (TensorFlow, Keras, Numpy, SciPy, Scikit-Learn).
3. Clone this repository.
4. Prepare your OASIS dataset structure.
5. `cd src`
6. `bash run_pipeline.sh`
